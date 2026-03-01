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
    # ('H', (0., 0., 9.0)), 
    # ('H', (0., 0.,10.0)),
    # ('H', (0., 0.,11.0)), 
    # ('H', (0., 0.,12.0))
    ]

# geom = [
#     ('H', (0., 0., 1.0)), 
#     ('O', (0., 0., 2.0)),
#     ('H', (0., 0., 3.0)), 
#     ]

# geom = [
#     ('N', (0., 0., 1.0)), 
#     ('N', (0., 0., 2.0)),
#     ]

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
    build_qb_ham = False,
    store_mo_ints=True,
    store_mo_ints_np=True,
    build_df_ham=0,
    df_icut=1.0e-6
    )
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

print("\n")
print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")
print("\n")

timer = qf.local_timer()

fci_comp1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

# csv_comp1 = qf.CUSVComputer(nel=nel, sz=sz, norb=norb)

csv_comp1 = qf.CUSVComputer(
    nel=nel, 
    sz=sz, 
    norb=norb,
    on_gpu=False,
    dtype=np.complex128,
    )

fci_comp1.hartree_fock()
csv_comp1.hartree_fock()

sqham = mol.sq_hamiltonian

hermitian_pairs = qf.SQOpPool()
hermitian_pairs.add_hermitian_pairs(1.0, sqham)

# print(f"sq ham: {sqham}")

app_sqop = True
app_tens = True
get_exp = True
app_exact_evo = False
app_trot = True



# ===> apply sqop <====

if(app_sqop):
    timer.reset()
    fci_comp1.apply_sqop(sqham)
    timer.record("FCI apply sqop")

    csv_comp1.to_gpu()

    timer.reset()
    csv_comp1.apply_sqop(sqham)
    timer.record("CuSV apply sqop")

    csv_comp1.to_cpu()

    dC = csv_comp1.get_fci_comp_state_diff(fci_comp1, do_phase_compare=True)
    print(f"post H app (exp) |dC|: {dC}")




# ===> apply tensor <====

if(app_tens):
    fci_comp1.hartree_fock()

    timer.reset()
    fci_comp1.apply_tensor_spat_012bdy(
        mol.nuclear_repulsion_energy, 
        mol.mo_oeis, 
        mol.mo_teis, 
        mol.mo_teis_einsum, 
        norb)
    timer.record('FCI apply tensor')

    dC = csv_comp1.get_fci_comp_state_diff(fci_comp1, do_phase_compare=True)
    print(f"post H app (tensor) |dC|: {dC}")


# ===> get exp val <====

if(get_exp):
    fci_comp1.hartree_fock()
    csv_comp1.hartree_fock()

    timer.reset()
    Efci = fci_comp1.get_exp_val_tensor(
        mol.nuclear_repulsion_energy, 
        mol.mo_oeis, 
        mol.mo_teis, 
        mol.mo_teis_einsum, 
        norb)
    timer.record('FCI tensor exp val')

    csv_comp1.to_gpu()

    timer.reset()
    Ecsv = csv_comp1.get_exp_val_opt(sqham)
    timer.record("CuSV exp val")

    print(f"Efci: {Efci}")
    print(f"Ecsv: {Ecsv}")

    # print(fci_comp1)
    # print(fqe_comp1)


# # ===> evovle exactly <====

# if(app_exact_evo):
#     fci_comp1.hartree_fock()
#     fqe_comp1.hartree_fock()

#     timer.reset()

#     fci_comp1.evolve_tensor_taylor(
#         mol.nuclear_repulsion_energy, 
#         mol.mo_oeis, 
#         mol.mo_teis, 
#         mol.mo_teis_einsum, 
#         norb,
#         time,
#         1.0e-15,
#         100,
#         False)

#     timer.record(f"FCI exact step")

#     Cfci1 = fci_comp1.get_state_deep()

#     timer.reset()

#     fqe_comp1.evolve_tensor_taylor(
#         mol.nuclear_repulsion_energy,
#         mol.mo_oeis_np,
#         mol.mo_teis_np,
#         time,
#         1.0e-15, # was 1.0e-15
#         100,
#         False)

#     timer.record(f"FQE exact step")
#     Cfqe1 = fqe_comp1.get_state_deep()
#     print(f"\n |Cfci| evolve exact: {Cfci1.norm()} \n")
#     print(f"\n |Cfqe| evolve exact: {np.linalg.norm(Cfqe1)} \n")
#     print(f"\n |dC1| evolve exact: {fqe_comp1.get_tensor_diff(Cfci1)} \n")



# # ===> evovle pool trotter <====

dtime = 0.20
r = 1
order = 1

N = 1

ah = False
adj = False

print("\n Time Evo Settings")
print("===========================")
print(f" time:      {dtime}")
print(f" r:         {r}")
print(f" order:     {order}")
print(f" antiherm:  {ah}")
print(f" adjoint:   {adj}")
print("\n")

fci_comp1.hartree_fock()
csv_comp1.hartree_fock()

# print(f"dtime: {dtime / max(1, int(r))}")

if(app_trot):
    
    for i in range(N):
        timer.reset()

        fci_comp1.evolve_pool_trotter(
            hermitian_pairs,
            dtime,
            r,
            order,
            antiherm=ah,
            adjoint=adj)

        timer.record(f"FCI Trotter step {i}")

        if not csv_comp1.on_gpu():
            csv_comp1.to_gpu()

        timer.reset()

        # YOU ARE HERE!
        csv_comp1.evolve_pool_trotter(
            hermitian_pairs,
            dtime,
            r,
            order,
            antiherm=ah,
            adjoint=adj)

        timer.record(f"CUSV Trotter step {i}")

        csv_comp1.to_cpu()

        dC = csv_comp1.get_fci_comp_state_diff(fci_comp1, do_phase_compare=True)
        print(f"post trotter step {i} |dC|: {dC}")

# print(fci_comp1)
# print(csv_comp1)


    # ===> evovle pool basic <====

    ah = True
    adj = False

    print("\n Time Evo Settings")
    print("===========================")
    print(f" time:      {1.0}")
    print(f" r:         {1}")
    print(f" order:     {1}")
    print(f" antiherm:  {ah}")
    print(f" adjoint:   {adj}")
    print("\n")


    sd_pool = qf.SQOpPool()
    sd_pool.set_orb_spaces(ref)
    sd_pool.fill_pool("SD")
    # print(sd_pool)

    fci_comp1.hartree_fock()
    csv_comp1.hartree_fock()


    timer.reset()

    fci_comp1.evolve_pool_trotter_basic(
        sd_pool,
        antiherm=ah,
        adjoint=adj)

    timer.record(f"FCI Trotter basic inplace")

    timer.reset()

    csv_comp1.evolve_pool_trotter_basic(
        sd_pool,
        antiherm=ah,
        adjoint=adj)

    timer.record(f"CuSV Trotter basic")

    csv_comp1.to_cpu()

    dC = csv_comp1.get_fci_comp_state_diff(fci_comp1, do_phase_compare=True)
    print(f"post UCC trotter step {i} |dC|: {dC}")



print(f" N hp's:     {len(hermitian_pairs.terms())}")


print("\n\n")
print(timer)

# print("\n\n")
# csv_comp1.print_timer()

# OMP_NUM_THREADS=4 OMP_PROC_BIND=TRUE OMP_DYNAMIC=FALSE \
# OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
# python sandbox/fqe_api/test_advanced_api_v1.py
# H10? on Nick's MacBook Pro

#              Process name                 Time (s)                  Percent
#             =============            =============            =============
#            FCI apply sqop                   0.7209                     1.71
#            FQE apply sqop                   0.4135                     0.98
#          FCI apply tensor                   0.0947                     0.22
#          FQE apply tensor                   0.0553                     0.13
#            FCI exact step                   4.1651                     9.86
#            FQE exact step                   1.0557                     2.50
#          FCI Trotter step                   5.6837                    13.45
#  FCI Trotter step inplace                   4.3180                    10.22
#          FQE Trotter step                  24.5347                    58.07
# FCI Trotter basic inplace                   0.1675                     0.40
#         FQE Trotter basic                   1.0389                     2.46

#                Total Time                  42.2479                   100.00