import numpy as np
import qforte as qf
from qiskit.circuit import QuantumCircuit
from qforte.helper.df_ham_helper import *
from qforte.utils.exponentiate import exponentiate_pauli_string

def trotter_evolve(mol, dt, N=1, r=1, order=1):

    ref = mol.hf_reference   #hartree fock reference state
    nel = sum(ref)           #num of electrons
    nqubits = len(ref)

    gphase = np.exp(-1.0j*dt*mol.nuclear_repulsion_energy) #not sure why we do this?

    ################################
    # TROTTERIZED HAMILTONIAN INIT #
    ################################

    #get the hamiltonian
    sqham = mol.sq_hamiltonian 

    #extract the hermitian pairs
    hermitian_pairs = qf.SQOpPool()
    hermitian_pairs.add_hermitian_pairs(1.0, sqham)

    # Initialize a circuit to hold the pauli strings and a float to hold the global phase
    trotter_circ = qf.Circuit()

    trotter_phase = 1.0
    for pair in hermitian_pairs.terms():
        #unpack the pair
        coeff = pair[0]
        sqop = pair[1]

        #transform the hermitian pair to a linear combination of pauli strings
        qbop = sqop.jw_transform()
        #reapply the coefficient (doublecheck you understand this)
        qbop.mult_coeffs(coeff)

        #exponentiate each pauli string and multiply them all together
        for term in qbop.terms():
            #exponentiate the string (as the time evolution operator)
            exp_op = exponentiate_pauli_string(-1j*dt*term[0], term[1])
            #left-multiply the final circuit to create the product of exponentiated pauli strings
            trotter_circ.add_circuit(exp_op[0])
            #extract the phases to a single prefactor
            trotter_phase *= exp_op[1]
            
        
    ######################
    # FOCK COMPUTER INIT #
    ######################

    #initialize fock computer
    c = qf.Computer(nqubits)

    #and set it to the hartree fock state
    c.hartree_fock(nel) 

    for i in range(1):

        #do time evolution on the fock computer
        c.apply_circuit(trotter_circ)

    c.scale(gphase)

    return (trotter_circ, c)


def hartree_fock(mol, circuit: QuantumCircuit):
    nel = sum(mol.hf_reference)  # number of electrons
    hf = QuantumCircuit(circuit.num_qubits)
    for i in range(nel):
        hf.x(i)
    return hf.compose(circuit)
