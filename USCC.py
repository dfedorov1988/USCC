# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from pathlib import Path
from qiskit_nature.drivers import PySCFDriver, UnitsType
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
import os
import numpy as np
import time
from pyscf import gto, scf, ao2mo
from qiskit import Aer
from qiskit_nature.mappers.second_quantization import ParityMapper, JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit.algorithms.optimizers import L_BFGS_B, SLSQP
from qiskit_nature.circuit.library import HartreeFock, UCC
from qiskit.algorithms import NumPyMinimumEigensolver
from scipy.optimize import minimize
from optimization_routines import VQE_expm
import h5py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def custom_excitation_list(num_spin_orbitals,
                           num_particles):
    # for now just plugging in excitation list computed outside
    my_excitation_list = custom_list

    return my_excitation_list


def compute_number_of_n_exc(list_of_exc, order):
    n = 0
    for exc in list_of_exc:
        if len(exc[0]) == order:
            n += 1
    return n


# %%
r = 1.8  # internuclear distance
basis = 'sto-3g'
max_exc_level = 4  # maximum excitation level, from 1 to 4
# thresholds are values for USCC coefficients at each iteration
thresholds = [0.04, 0.02, 0.01, 0.005, 0.002,
              0.001, 0.0005, 0.0002, 1e-4, 1e-5, 1e-6, 1e-7]

print("Only computing excitations up to order", max_exc_level)

# Uncomment desired molecule below
# filename sets the hdf5 file where all results are saved for future analysis

# H4 linear test
atom = 'H 0 0 0; H 0 0 {}; H 0 0 {}; H 0 0 {}'.format(r, r*2, r*3)
filename = "test.hdf5"

# # H6 linear symmetric
# atom = 'H 0 0 0; H 0 0 {}; H 0 0 {}; H 0 0 {}; H 0 0 {}; H 0 0 {}'.format(
#     r, r*2, r*3, r*4, r*5)
# filename = "h6_lin_sym_loc_r18.hdf5"

# # H6 linear non-symmetric
# atom='H 0 0 0; H 0 0 {}; H 0 0 {}; H 0 0 {}; H 0 0 {}; H 0 0 {}'.format(r, r*2, r*3, r*4, r*5-0.15)
# filename = "h6_lin_nonsym_r22.hdf5"

# # BeH2 symmetric
# atom = 'Be .0 .0 .0; H .0 .0 {}; H .0 .0 {}'.format(r, -r)
# filename = "beh2_sym_r13.hdf5"

# BeH2 non-symmetric
# atom = 'Be .0 .0 .0; H .0 .0 {}; H .0 .0 {}'.format(r, -0.9*r)
# filename = "beh2_nonsym_r13.hdf5"

# H2O
# theta = 104.5
# c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
# R = np.array(((c, -s), (s, c)))  # Rotation matrix
# x, y = R @ np.array((0, r))

# # H2O symmetric
# atom = 'O .0 .0 .0; H .0 .0 {}; H .0 {} {}'.format(r, x, y)
# filename = "h2o_sym_r24.hdf5"

# H2O non-symmetric
# atom='O .0 .0 .0; H .0 .0 {}; H .0 {} {}'.format(r, x, 0.9*y)
# filename = "h2o_nonsym_r24.hdf5"

print("Molecule:\n", atom)

# %%
# For pure PySCF calculations
mol = gto.M(atom=atom, basis=basis)
nelec = mol.nelectron
mf = scf.RHF(mol)
e_hf = mf.kernel()
h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
eri = ao2mo.kernel(mol, mf.mo_coeff)
norb = h1.shape[1]

# %%
# Classical PySCF calculations
driver = PySCFDriver(atom=atom,
                     unit=UnitsType.ANGSTROM,
                     basis=basis)
problem = ElectronicStructureProblem(driver)
molecule = driver.run()
h1 = molecule.one_body_integrals
h2 = molecule.two_body_integrals
e_nr = molecule.nuclear_repulsion_energy
print("Nuclear repulsion energy:", e_nr)

# %%
# Preparing for VQE simulations
second_q_ops = problem.second_q_ops()
main_op = second_q_ops[0]
num_particles = (problem.molecule_data_transformed.num_alpha,
                 problem.molecule_data_transformed.num_beta)
num_spin_orbitals = 2 * problem.molecule_data.num_molecular_orbitals
optimizer = SLSQP(disp=True)
mapper = JordanWignerMapper()
converter = QubitConverter(mapper=mapper, two_qubit_reduction=False)
qubit_op = converter.convert(main_op, num_particles=num_particles)
backend = Aer.get_backend('statevector_simulator')
num_qubits = qubit_op.num_qubits
init_state = HartreeFock(num_qubits, num_particles, converter)
# %%
#### Exact solution ####
numpy_solver = NumPyMinimumEigensolver()
ret_exact = numpy_solver.compute_minimum_eigenvalue(qubit_op)
print("Exact energy:", ret_exact._eigenvalue.real)
# %%
# generating UCCSD ansatz
ansatz_sd = UCC(num_spin_orbitals=qubit_op.num_qubits, num_particles=num_particles,
                qubit_converter=converter, excitations='sd', initial_state=init_state, reps=1)
singles_doubles = ansatz_sd._get_excitation_list()
ansatz_sd._build()

# %%
# Run standard UCCSD using exponentiation
algorithm_sd = VQE_expm(ansatz_sd, qubit_op)
initial_point = np.zeros(ansatz_sd.num_parameters)
print("\nStarting VQE with {} parameters".format(ansatz_sd.num_parameters))
result = minimize(algorithm_sd.cost_fn, initial_point, method="slsqp")
print("Error:", result.fun - ret_exact.eigenvalue.real)
uccsd_n_params = ansatz_sd.num_parameters
e_uccsd = result.fun

# %%
# Generating dictionary containing single and double excitations
exc_dict = dict()
for exc in singles_doubles:
    if len(exc[0]) == 1:
        exc_dict[exc] = {'h1': h1[exc[0], exc[1]]}
    if len(exc[0]) == 2:
        exc_dict[exc] = {'h2': h2[exc[1][1], exc[1][0], exc[0][0], exc[0][1]]}
# %%
def run_iter(thresh, prev_n_params, initial_guess):
    """main loop for the iterative procedure with selecting 
    using h[i,a], h[i,j,a,b], t1[i,a], t2[i,j,a,b]"""
    for exc in exc_dict.keys():
        if exc not in custom_list:
            if exc_dict[exc]:
                # picking max value as a importance coefficient for each excitation
                max_value = np.max(np.abs(list(exc_dict[exc].values())))
            else:
                continue

            if len(exc[0]) <= max_exc_level:  # filtering to desired highest level of excitation
                if max_value > thresh:
                    exc_set = set(
                        [element for tupl in exc for element in tupl])
                    if exc_set not in sets_total:
                        custom_list.append(exc)
                        sets_total.append(exc_set)

    print("Number of operators in ansatz:", len(custom_list))
    params_dict = dict()
    ansatz = UCC(num_spin_orbitals=qubit_op.num_qubits, num_particles=num_particles,
                 qubit_converter=converter, excitations=custom_excitation_list,
                 initial_state=init_state)

    if prev_n_params < len(custom_list):
        thresholds_plot.append(thresh)
        ansatz._build()  # Make the operators list
        algorithm = VQE_expm(ansatz, qubit_op)
        add_zeros = np.zeros(ansatz.num_parameters - len(initial_guess))
        initial_point = np.append(initial_guess, add_zeros)
        print("\nStarting VQE with {} parameters".format(len(custom_list)))
        print("Singles:", compute_number_of_n_exc(custom_list, 1))
        print("Doubles:", compute_number_of_n_exc(custom_list, 2))
        print("Triples:", compute_number_of_n_exc(custom_list, 3))
        print("Quadruples:", compute_number_of_n_exc(custom_list, 4))
        result = minimize(algorithm.cost_fn, initial_point,
                          method="slsqp", options={'disp': True})
        n_evals = result.nfev
        n_iter = result.nit
        n_iter_array.append(n_iter)
        n_evals_array.append(n_evals)
        results.append(result)
        print("Error", (result.fun - ret_exact._eigenvalue.real)*1000, "mHa")
        prev_n_params = ansatz.num_parameters
        for n in range(len(result.x)):
            exc = custom_list[n]
            exc_dict[exc]['opt_coeff'] = result.x[n]

        # h1t1
        tmp_keys = list(exc_dict.keys())
        for exc_opt in custom_list:
            for exc1 in tmp_keys:
                if len(exc_opt[0]) == 1 and len(exc1[0]) == 1:
                    exc2 = tuple([tuple(sorted([exc_opt[0][0], exc1[0][0]])),
                                  tuple(sorted([exc_opt[1][0], exc1[1][0]]))])
                    exc2_set = set(
                        [element for tupl in exc2 for element in tupl])
                    if len(exc2_set) < 4:
                        continue
                    if exc2 in exc_dict.keys():
                        exc_dict[exc2]['t1h1'] = exc_dict[exc_opt]['opt_coeff'] * \
                            exc_dict[exc1]['h1']
                        if "opt_coeff" in exc_dict[exc1].keys():
                            exc_dict[exc2]['h1t1'] = exc_dict[exc_opt]['h1'] * \
                                exc_dict[exc1]['opt_coeff']

        # h1t2, t1h2
        tmp_keys = list(exc_dict.keys())
        for exc_opt in custom_list:
            for exc2 in tmp_keys:
                if len(exc_opt[0]) == 1 and len(exc2[0]) == 2:
                    exc3 = tuple([tuple(sorted([exc_opt[0][0], exc2[0][0], exc2[0][1]])),
                                  tuple(sorted([exc_opt[1][0], exc2[1][0], exc2[1][1]]))])
                    exc3_set = set(
                        [element for tupl in exc3 for element in tupl])
                    if len(exc3_set) < 6:
                        continue
                    t1h2 = exc_dict[exc_opt]['opt_coeff'] * \
                        exc_dict[exc2]['h2']
                    if exc3 in exc_dict.keys():
                        if 't1h2' in exc_dict[exc3].keys():
                            exc_dict[exc3]['t1h2'] = np.max(
                                np.abs([t1h2, exc_dict[exc3]['t1h2']]))
                        else:
                            exc_dict[exc3]['t1h2'] = t1h2
                        if "opt_coeff" in exc_dict[exc2].keys():
                            exc_dict[exc3]['h1t2'] = exc_dict[exc_opt]['h1'] * \
                                exc_dict[exc2]['opt_coeff']
                    if exc3 not in exc_dict.keys():
                        exc_dict[exc3] = {'t1h2': t1h2}
                        if "opt_coeff" in exc_dict[exc2].keys():
                            exc_dict[exc3]['h1t2'] = exc_dict[exc_opt]['h1'] * \
                                exc_dict[exc2]['opt_coeff']

        # t2h1, h2t1
        for exc_opt in custom_list:
            for exc1 in tmp_keys:
                if len(exc_opt[0]) == 2 and len(exc1[0]) == 1:
                    exc3 = tuple([tuple(sorted([exc_opt[0][0], exc_opt[0][1], exc1[0][0]])),
                                  tuple(sorted([exc_opt[1][0], exc_opt[1][1], exc1[1][0]]))])
                    exc3_set = set(
                        [element for tupl in exc3 for element in tupl])
                    if len(exc3_set) < 6:
                        continue
                    t2h1 = exc_dict[exc_opt]['opt_coeff'] * \
                        exc_dict[exc1]['h1']
                    if exc3 in exc_dict.keys():
                        if 't2h1' in exc_dict[exc3].keys():
                            exc_dict[exc3]['t2h1'] = np.max(
                                np.abs([t2h1, exc_dict[exc3]['t2h1']]))
                        else:
                            exc_dict[exc3]['t2h1'] = t2h1
                        if "opt_coeff" in exc_dict[exc1].keys():
                            exc_dict[exc3]['h2t1'] = exc_dict[exc_opt]['h2'] * \
                                exc_dict[exc1]['opt_coeff']

                    if exc3 not in exc_dict.keys():
                        exc_dict[exc3] = {'t2h1': t2h1}
                        if "opt_coeff" in exc_dict[exc1].keys():
                            exc_dict[exc3]['h2t1'] = exc_dict[exc_opt]['h2'] * \
                                exc_dict[exc1]['opt_coeff']

        # t2h2
        for exc_opt in custom_list:
            for exc2 in tmp_keys:
                if len(exc_opt[0]) == 2 and len(exc2[0]) == 2:
                    exc4 = tuple([tuple(sorted([exc_opt[0][0], exc_opt[0][1], exc2[0][0], exc2[0][1]])),
                                  tuple(sorted([exc_opt[1][0], exc_opt[1][1], exc2[1][0], exc2[1][1]]))])
                    exc4_set = set(
                        [element for tupl in exc4 for element in tupl])
                    if len(exc4_set) < 8:
                        continue
                    t2h2 = exc_dict[exc_opt]['opt_coeff'] * \
                        exc_dict[exc2]['h2']
                    if exc4 in exc_dict.keys():
                        if 't2h2' in exc_dict[exc4].keys():
                            exc_dict[exc4]['t2h2'] = np.max(
                                np.abs([t2h2, exc_dict[exc4]['t2h2']]))
                        else:
                            exc_dict[exc4]['t2h2'] = t2h2
                        if "opt_coeff" in exc_dict[exc2].keys():
                            exc_dict[exc4]['h2t2'] = exc_dict[exc_opt]['h2'] * \
                                exc_dict[exc2]['opt_coeff']

                    if exc4 not in exc_dict.keys():
                        exc_dict[exc4] = {'t2h2': t2h2}
                        if "opt_coeff" in exc_dict[exc2].keys():
                            exc_dict[exc4]['h2t2'] = exc_dict[exc_opt]['h2'] * \
                                exc_dict[exc2]['opt_coeff']

        optimized_params = result.x
        for n in range(len(optimized_params)):
            params_dict[custom_list[n]] = optimized_params[n]
    else:
        print("Threshold results in the same number of parameters, skipping")
        optimized_params = initial_guess

    return prev_n_params, optimized_params, params_dict


# %%
# running USCC for thresholds set in the beginning of the file
sets_total = []
t0 = time.time()
thresholds_plot = []
custom_list = []
results = []
n_params = 0
n_evals_array = []
n_iter_array = []
optimized_coeffs = [0]
parameters_array = []
param_dicts = []

for thresh in thresholds:
    print("\nThreshold:", thresh)
    prev_n_params = n_params
    initial_guess = optimized_coeffs
    n_params, optimized_coeffs, params_dict = run_iter(
        thresh, prev_n_params, initial_guess)
    if len(params_dict.keys()) > 0:
        param_dicts.append(params_dict)
    parameters_array.append(optimized_coeffs.copy())

t1 = time.time()
total_time = t1 - t0

# %%
# printing data for each iteration
n = 0
prev_e = 0
n_params = []
print("{:>14s}{:>14s}{:>14s}{:>8s}{:>8s}{:>8s}".format("Threshold", "Error, mHa",
                                                       "dE, mHa", "Ntotal", "Niter", "Nevals"))
for result in results:
    e = result.fun
    print("{:>14.4e}{:>14.4e}{:>14.4e}{:>8d}{:>8d}{:>8d}".format(
        thresholds[n], (result.fun-ret_exact.eigenvalue.real)*1000,
        (result.fun-prev_e)*1000, len(result.x), n_iter_array[n], n_evals_array[n]))
    prev_e = result.fun
    n_params.append(len(result.x))
    n += 1

# %%
# All data saved into a structured hdf5 file for future analysis and plotting
full_name = Path("hdf5/" + filename)
if full_name.is_file():
    print("Filename already exists")
    os.rename("hdf5/" + filename, "hdf5/" + filename + ".prev")
    print("Renamed the old file with .prev")
f = h5py.File("hdf5/" + filename, "a")
f.attrs['e_exact'] = ret_exact.eigenvalue.real
f.attrs['e_nr'] = e_nr
f.attrs['atom'] = str(atom)
f.attrs['e_uccsd'] = e_uccsd
f.attrs['uccsd_n_params'] = uccsd_n_params

for n in range(len(thresholds_plot)):
    grp = f.create_group("iter_" + str(n+1))
    grp.create_group("opt_params")
    grp.attrs['n_iter'] = n_iter_array[n]
    grp.attrs['n_evals'] = n_evals_array[n]
    grp.attrs['e'] = results[n].fun
    grp.attrs['thresh'] = thresholds_plot[n]
    # writing excitations and corresponding optimized parameters
    for key in param_dicts[n].keys():
        grp['opt_params'].attrs[str(key)] = param_dicts[n][key]
f.close()
print("Done")
