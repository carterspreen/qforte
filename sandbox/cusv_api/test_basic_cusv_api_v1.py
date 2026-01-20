import qforte as qf
import numpy as np
 
import time

# Define the reference and geometry lists.
geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    ('H', (0., 0., 5.0)), 
    ('H', (0., 0., 6.0)),
    ('H', (0., 0., 7.0)), 
    ('H', (0., 0., 8.0)),
    ('H', (0., 0., 9.0)), 
    ('H', (0., 0.,10.0)),
    # ('H', (0., 0.,11.0)), 
    # ('H', (0., 0.,12.0))
    ]

geom = [
    ('N', (0., 0., 1.0)), 
    ('N', (0., 0., 2.0)),
    ]

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
# mol = qf.system_factory(
#     build_type='psi4', 
#     mol_geometry=geom, 
#     basis='sto-3g', 
#     run_fci=1)

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g', 
    run_fci=0,
    # build_qb_ham = False,
    store_mo_ints=0,
    build_df_ham=0,
    df_icut=1.0e-6
    )
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")

real = False

if real:
    data_type_str = "real"
    dtype = np.float64
else:
    data_type_str = "complex"
    dtype = np.complex128
 
fci_comp1 = qf.FCIComputerGPU(
    nel=nel, 
    sz=sz, 
    norb=norb,
    on_gpu=False,
    data_type=data_type_str,
    )


csv_comp1 = qf.CUSVComputer(
    nel=nel, 
    sz=sz, 
    norb=norb,
    on_gpu=False,
    dtype=dtype,
    )


# Try to devic
fci_comp1.to_gpu()
csv_comp1.to_gpu()

# Try back to host
fci_comp1.to_cpu()
csv_comp1.to_cpu()

# print(f"fci on cpu: {fci_comp1.on_cpu()}")
# print(f"csv on cpu: {csv_comp1.on_cpu()}")

# Test HF construction
fci_comp1.hartree_fock_cpu()
csv_comp1.hartree_fock()

# Test setter
fci_comp1.set_element([0,1], 2.0)
csv_comp1.set_element_from_IaIb([0,1], 2.0)


# Test state getter
Na = fci_comp1.get_Na()
Nb = fci_comp1.get_Nb()

C1 = qf.Tensor([Na, Nb], "C3")
fci_comp1.copy_to_tensor_cpu(C1)
C2 = csv_comp1.get_state()

print(f"Type C1: {type(C1)}")
print(f"Type C2: {type(C2)}")

# print("\n ===> Printing returned Tensor/nparray from get_state() <===")
# print(C1)
# print("\n ===> numpy version <===")
# print(C2)
# print("\n")

# Test printing functions (via __str__ method)
# print(fci_comp1)
# print(csv_comp1)
# print(fci_comp1.str(print_complex=True))
# print(csv_comp1.str(print_complex=True))

# # Test diff
# # fci_comp1.get_tensor_diff(C2)
# print("\n")
# print(f" Diff (fci - fqe): {fqe_comp1.get_tensor_diff(C1)}")
# print("\n")

# # Test getter and setter
fci_00 = fci_comp1.get_element([0,0])
csv_00 = csv_comp1.get_element_from_IaIb([0,0])

fci_01 = fci_comp1.get_element([0,1])
csv_01 = csv_comp1.get_element_from_IaIb([0,1])

print("\n")
print(f" fci_00: {fci_00}")
print(f" csv_00: {csv_00}")
print(f" fci_01: {fci_01}")
print(f" csv_01: {csv_01}")
print("\n")

# Test Diff
dC1 = csv_comp1.get_fci_comp_state_diff(fci_comp1)
print(f"post set |dC|: {dC1}")

# Test scale
fci_comp1.scale(-1.5)
csv_comp1.scale(-1.5)
# print(fci_comp1)
# print(csv_comp1)

dC2 = csv_comp1.get_fci_comp_state_diff(fci_comp1)
print(f"post scale |dC|: {dC2}")



# Test apply individual sqop..

sd_pool = qf.SQOpPool()
sd_pool.set_orb_spaces(ref)
sd_pool.fill_pool("SD")
# print(sd_pool)

M = len(sd_pool.terms())

# fci_comp1.hartree_fock_cpu()
# csv_comp1.hartree_fock()

# fci_comp1.to_gpu()
# csv_comp1.to_gpu()

# for mu in range(M):

#     fci_comp1.hartree_fock_cpu()
#     csv_comp1.hartree_fock()

#     fci_comp1.to_gpu()
#     csv_comp1.to_gpu()


#     # Think about how getting a specific term is done think there is more
#     # overhead with this than we want (is returning a copy presently)
#     op = sd_pool.terms()[mu][1]
#     print(f"\n\n ===> new op mu: {mu} <===")
#     print(op)

#     fci_comp1.apply_sqop_gpu(op)
#     csv_comp1.apply_sqop_v2(op)

#     fci_comp1.to_cpu()
#     csv_comp1.to_cpu()

#     # C2 = csv_comp1.get_state()

#     # print(fci_comp1)
#     # print(C2)

#     dC = csv_comp1.get_fci_comp_state_diff(fci_comp1, do_phase_compare=True)
#     print(f"mu: {mu} post scale |dC|: {dC}")

#     # fci_01 = fci_comp1.get_element([1,0])
#     # csv_01 = csv_comp1.get_element_from_IaIb([1,0])

#     # print("\n")
#     # print(f" fci_01: {fci_01}")
#     # print(f" csv_01: {csv_01}")
#     # print("\n")


# Try apply hamiltonain

tmr = qf.local_timer()

fci_comp1.hartree_fock_cpu()
csv_comp1.hartree_fock()

fci_comp1.to_gpu()
csv_comp1.to_gpu()


tmr.reset()
fci_comp1.apply_sqop_gpu(mol.sq_hamiltonian)
tmr.record("FCI apply H as sqop")

tmr.reset()
csv_comp1.apply_sqop(mol.sq_hamiltonian)
tmr.record("CSV apply H as sqop (rotation)")

fci_comp1.to_cpu()
csv_comp1.to_cpu()

dC = csv_comp1.get_fci_comp_state_diff(fci_comp1, do_phase_compare=True)
print(f"post H app (exp) |dC|: {dC}")

fci_comp1.to_gpu()
csv_comp1.to_gpu()

csv_comp1.hartree_fock()

tmr.reset()
csv_comp1.apply_sqop_v2(mol.sq_hamiltonian)
tmr.record("CSV apply H as sqop (palui mat)")

fci_comp1.to_cpu()
csv_comp1.to_cpu()

dC = csv_comp1.get_fci_comp_state_diff(fci_comp1, do_phase_compare=True)
print(f"post H app pauli |dC|: {dC}")


fci_comp1.hartree_fock_cpu()
csv_comp1.hartree_fock()

fci_comp1.to_gpu()
csv_comp1.to_gpu()

tmr.reset()
Efci = fci_comp1.get_exp_val(mol.sq_hamiltonian)
tmr.record("FCI Ham Exp")

tmr.reset()
Ecsv = csv_comp1.get_exp_val_opt(mol.sq_hamiltonian)
tmr.record("CSV Ham Exp")

dE = Efci - Ecsv
print(f"dE ham expectation val: {dE}")

print(tmr)






