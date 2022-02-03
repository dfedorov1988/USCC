# import common packages
import numpy as np
import warnings
from qiskit import Aer, execute
from qiskit.circuit import ParameterExpression
from qiskit.opflow.gradients import Gradient
from qiskit.opflow import StateFn, CircuitStateFn
import qiskit
import numbers

from qiskit import transpile
from subprocess import check_output, run, call, Popen, PIPE, STDOUT
from scipy.optimize import minimize
from qiskit.quantum_info import Pauli
import copy
from collections import Counter
from scipy.linalg import eig
from qiskit.circuit import parametervector, Parameter
import sys
import time
import scipy
sys.path.insert(0, '..')

class VQE_expm:

    def __init__(self, var_form, qubitOp):
        self.qubitOp = qubitOp
        self.qubitOp_mat = qubitOp.to_spmatrix()

        # Store all of the cluster operators as vform ops
        self.var_form_ops = copy.deepcopy(var_form.operators)
        for i in range(len(self.var_form_ops)):
            self.var_form_ops[i] = self.var_form_ops[i].to_spmatrix()

        self.var_form = var_form
        self.init_state = execute(var_form.initial_state, Aer.get_backend(
            "statevector_simulator")).result().get_statevector()

    def cost_fn(self, theta, *args):
        total_op = self.var_form_ops[0] * 0  # 0 op so we can add
        for this_theta, this_op in zip(theta, self.var_form_ops):
            total_op += this_op*this_theta

        u_wf = scipy.sparse.linalg.expm_multiply(
            -1.j*total_op, self.init_state)
        #u_wf = expm_multiply_parallel(-1.j*total_op).dot(self.init_state)
        H = self.qubitOp_mat

        e = np.real(np.conjugate(np.transpose(u_wf))@H@u_wf)
        return e


class VQE_qk:

    def __init__(self, var_form, qubitOp, backend_sim=None, n_samples=8192, param_scales=None):
        self.n_samples_default = n_samples
        self.qubitOp = qubitOp
        print("start VQE initialization")
        self.qubitOp_p = qubitOp.to_pauli_op()
        # Get sp_matrix and coeff arrays
        self.qubitOp_ind_mats = []
        self.qubitOp_ind_coeffs = []
        for pauli in self.qubitOp_p:
            self.qubitOp_ind_mats.append(pauli.primitive.to_spmatrix())
            self.qubitOp_ind_coeffs.append(pauli.coeff)
        print("Converting to sparse matrix")
        self.qubitOp_mat = qubitOp.to_spmatrix()
        self.var_form = var_form

        if(backend_sim is None):
            self.backend_sim = Aer.get_backend('statevector_simulator')
        else:
            self.backend_sim = backend_sim
        # print("Gradients")
        # self.grad_circs = Gradient(grad_type="parameter_shift").convert(
        #     CircuitStateFn(var_form), params=var_form.parameters.data)

    def cost_fn(self, theta, *args):
        t1 = time.time()
        energy = self._eval_circ(self.var_form, theta)
        dt = time.time() - t1
        print("Energy:", energy, "evaluated in", dt, "seconds")

        return energy

    def _eval_circ(self, circ, theta):
        value_dict = {}
        for i, param in enumerate(circ.parameters):
            value_dict[param] = theta[i]

        bcirc = circ.bind_parameters(value_dict)

        res = execute(bcirc, self.backend_sim).result()
        sv = res.get_statevector()
        print("STATEVECTOR", sv)
        fval = np.real(np.conjugate(np.transpose(sv))@self.qubitOp_mat@sv)

        return fval

    def _sample_binomial(self, sv, n_samples):
        stoch_evs = []  # stochastically estimated evs
        prefs = []  # prefactors
        p0s = []  # bernouilli probs
        shots = []  # individual sample results
        ev_tot = 0
        for i in range(len(self.qubitOp_ind_mats)):
            this_pref = self.qubitOp_ind_coeffs[i]
            prefs.append(this_pref)
            # Get the ev of this pauli.
            # It will be a number between -1,1, so we add 1 and subtract by 2 to get our bernouilli prob
            this_p0 = (np.real(np.conjugate(np.transpose(sv))
                       @ self.qubitOp_ind_mats[i]@sv)+1)/2
            p0s.append(this_p0)
            # sample from the binomial distribution
            # And shift to -1, 1
            if(this_p0 > 1):
                # Fix because sometimes it is 1+\epsilon
                this_p0 = 1

            this_shots = (np.random.binomial(1, this_p0, n_samples)-0.5)*2
            this_stoch_ev = np.sum(this_shots)/n_samples

            stoch_evs.append(this_stoch_ev)
            shots.append(this_shots)
            ev_tot += np.real(this_stoch_ev * this_pref)

        return ev_tot, np.array(stoch_evs), np.array(prefs), np.array(p0s), np.array(shots)

    def cost_fn_stoch(self, theta, n_samples=None):
        if(n_samples == None):
            n_samples = self.n_samples_default
        energy, sample_data = self._eval_circ_stoch(
            self.var_form, theta, n_samples)
        return energy

    def cost_fn_stoch_full(self, theta, n_samples=None):
        #########
        # Returns the cost fn, as well as a dictionary storing
        # various information about the sampling, including:
        # sample_data['stoch_evs']: stochastically estimated value of each element of the sum. Drawn from a binomial distribution, then shifted to [-1,1]. Add 1, divide by 2 to get observed p0
        #
        # sample_data['prefs'] = fixed prefactors multiplying stochastic evs above
        #
        # sample_data['p0s'] = actual p0 used in binomial sampling. Using this information is 'cheating', because it is not available on the real QC
        #########
        if(n_samples == None):
            n_samples = self.n_samples_default

        energy, sample_data = self._eval_circ_stoch(
            self.var_form, theta, n_samples)
        return energy, sample_data

    def _eval_circ_stoch(self, circ, theta, n_samples):
        value_dict = {}
        for i, param in enumerate(circ.parameters):
            value_dict[param] = theta[i]

        bcirc = circ.bind_parameters(value_dict)
        res = execute(bcirc, self.backend_sim).result()
        sv = res.get_statevector()

        energy, stoch_evs, prefs, p0s, shots = self._sample_binomial(
            sv, n_samples)

        sample_data = {}
        sample_data['stoch_evs'] = stoch_evs  # stochastically estimated evs
        sample_data['prefs'] = prefs
        sample_data['p0s'] = p0s
        sample_data['shots'] = shots

        return energy, sample_data

    def grad(self, theta):
        grad_val = np.zeros(len(theta))
        for i in range(len(theta)):
            grad_val[i] = self.grad_i(theta, i)
        return grad_val

    def _grad_two_terms(self, theta, i, two_terms_op):
        # Only two values assumed
        grad_val_i = two_terms_op[0].coeff * \
            self._eval_circ(two_terms_op[0].primitive, theta)
        # Subtract second term
        grad_val_i -= two_terms_op[1].coeff * \
            self._eval_circ(two_terms_op[1].primitive, theta)
        return grad_val_i

    def grad_i(self, theta, i):
        # Get gradient with respect to argument i

        if(type(self.grad_circs[i]) is qiskit.opflow.list_ops.summed_op.SummedOp):
            # a SummedOp has several ListOps; we loop through them
            grad_val_i = 0
            for j in range(len(self.grad_circs[i])):
                grad_val_i += self.grad_circs[i][j].coeff/2 * \
                    self._grad_two_terms(theta, i, self.grad_circs[i][j])

        else:
            # Only one 'ListOp' -> only one pair
            grad_val_i = self.grad_circs[i].coeff/2. * \
                self._grad_two_terms(theta, i, self.grad_circs[i])

        return grad_val_i

    def _grad_two_terms_stoch(self, theta, i, two_terms_op, n_samples):
        # Only two values assumed
        grad_val_i_0, sample_data_0 = self._eval_circ_stoch(
            two_terms_op[0].primitive, theta, n_samples)
        # Subtract second term
        grad_val_i_1, sample_data_1 = self._eval_circ_stoch(
            two_terms_op[1].primitive, theta, n_samples)
        grad_val_i = two_terms_op[0].coeff * \
            grad_val_i_0 - two_terms_op[1].coeff * grad_val_i_1
        return grad_val_i, sample_data_0, sample_data_1

    def grad_stoch(self, theta, n_samples=None):
        if(n_samples == None):
            n_samples = self.n_samples_default

        grad_val = np.zeros(len(theta))
        for i in range(len(theta)):
            grad_val[i] = self.grad_i_stoch(theta, i, n_samples)

        return grad_val

    def grad_i_stoch(self, theta, i, n_samples=None):
        if(n_samples == None):
            n_samples = self.n_samples_default

        grad_val_i, _ = self.grad_i_stoch_full(theta, i, n_samples)

        return grad_val_i

    def grad_i_stoch_full(self, theta, i, n_samples=None):
        if(n_samples == None):
            n_samples = self.n_samples_default

        #########
        # Returns the gradient for parameter 'i', as well as a dictionary storing
        # various information about the sampling, including for each circuit
        # sample_data['stoch_evs']: stochastically estimated value of each element of the sum. Drawn from a binomial distribution, then shifted to [-1,1]. Add 1, divide by 2 to get observed p0
        #
        # sample_data['prefs'] = fixed prefactors multiplying stochastic evs above
        #
        # sample_data['p0s'] = actual p0 used in binomial sampling. Using this information is 'cheating', because it is not available on the real QC
        #
        # sample_data['shots'] = individual -1 or 1 shots generated by the sampling
        #
        ##########

        # Get gradient with respect to argument i
        sample_datas = []
        grad_val_i = 0
        if(type(self.grad_circs[int(i)]) is qiskit.opflow.list_ops.summed_op.SummedOp):
            # a SummedOp has several ListOps; we loop through them
            grad_val_i = 0
            for j in range(len(self.grad_circs[i])):
                this_grad_val_i, sample_data_0, sample_data_1 = self._grad_two_terms_stoch(
                    theta, i, self.grad_circs[i][j], n_samples)

                grad_val_i += self.grad_circs[i][j].coeff/2 * this_grad_val_i
                sample_data_0["mult"] = self.grad_circs[i][j][0].coeff * \
                    self.grad_circs[i][j].coeff/2
                sample_data_1["mult"] = self.grad_circs[i][j][1].coeff * \
                    self.grad_circs[i][j].coeff/2

                sample_datas.append(sample_data_0)
                sample_datas.append(sample_data_1)
        else:
            # Only one 'ListOp' -> only one pair
            this_grad_val_i, sample_data_0, sample_data_1 = self._grad_two_terms_stoch(
                theta, i, self.grad_circs[i], n_samples)
            grad_val_i = self.grad_circs[i].coeff/2 * this_grad_val_i
            sample_data_0["mult"] = self.grad_circs[i][0].coeff * \
                self.grad_circs[i].coeff/2
            sample_data_1["mult"] = self.grad_circs[i][1].coeff * \
                self.grad_circs[i].coeff/2

            sample_datas.append(sample_data_0)
            sample_datas.append(sample_data_1)

        return grad_val_i, sample_datas

    def grad_stoch_full(self, theta, n_samples=None):
        if(n_samples == None):
            n_samples = self.n_samples_default

        grad_val = np.zeros(len(theta))
        sample_datas = []

        for i in range(len(theta)):
            grad_i, sample_datas_i = self.grad_i_stoch_full(
                theta, i, n_samples)
            grad_val[i] = grad_i
            sample_datas.append(sample_datas_i)

        return grad_val, sample_datas

    def get_Ls(self):
        Ls = []
        for i in range(len(self.grad_circs)):
            mults = []

            if(type(self.grad_circs[i]) is qiskit.opflow.list_ops.summed_op.SummedOp):
                # a SummedOp has several ListOps; we loop through them
                for j in range(len(self.grad_circs[i])):
                    # Now we have 2 terms, get their mults
                    mults.append(
                        self.grad_circs[i][j][0].coeff*self.grad_circs[i][j].coeff/2)
                    mults.append(
                        self.grad_circs[i][j][1].coeff*self.grad_circs[i][j].coeff/2)
            else:
                # Only one 'ListOp' -> only one pair
                # Now we have 2 terms, get their mults
                mults.append(
                    self.grad_circs[i][0].coeff*self.grad_circs[i].coeff/2)
                mults.append(
                    self.grad_circs[i][1].coeff*self.grad_circs[i].coeff/2)

            Ls.append(np.sum(np.multiply(np.abs(mults),
                      np.sum(np.abs(self.qubitOp_ind_coeffs)))))

        return Ls

    def get_nj1s(self):
        nj1s = []
        for i in range(len(self.grad_circs)):
            mults = []

            if(type(self.grad_circs[i]) is qiskit.opflow.list_ops.summed_op.SummedOp):
                # a SummedOp has several ListOps; we loop through them
                for j in range(len(self.grad_circs[i])):
                    # Now we have 2 terms, get their mults
                    mults.append(
                        self.grad_circs[i][j][0].coeff*self.grad_circs[i][j].coeff/2)
                    mults.append(
                        self.grad_circs[i][j][1].coeff*self.grad_circs[i][j].coeff/2)
            else:
                # Only one 'ListOp' -> only one pair
                # Now we have 2 terms, get their mults
                mults.append(
                    self.grad_circs[i][0].coeff*self.grad_circs[i].coeff/2)
                mults.append(
                    self.grad_circs[i][1].coeff*self.grad_circs[i].coeff/2)

            nj1s.append(len(mults))

        return nj1s
