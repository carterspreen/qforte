import numpy as np
from qforte import Circuit
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

def qforte_to_qiskit(qforte_circuit: Circuit, nqubits: int) -> QuantumCircuit:
    '''Takes a qForte circuit object and returns 
    an equivalent qiskit QuantumCircuit object'''
    qiskit_circuit = QuantumCircuit(nqubits)
    gates = qforte_circuit.gates()

    # Rzy_mat = 1/np.sqrt(2) * np.array([[1j, 1], [1, 1j]])

    gate_map_1qbit = {
        "X": qiskit_circuit.x,
        "Y": qiskit_circuit.y,
        "Z": qiskit_circuit.z,
        "H": qiskit_circuit.h,
        "V": qiskit_circuit.sx,
        "S": qiskit_circuit.s,
        "T": qiskit_circuit.t,
        "I": qiskit_circuit.id,
        # "Rzy": qiskit_circuit.unitary,
        "Rzy": lambda target: qiskit_circuit.rx(np.pi/2, target),
        "adj(Rzy)": lambda target: qiskit_circuit.rx(-np.pi/2, target),
        # "adj(Rzy)": qiskit_circuit.unitary
    }

    gate_map_2qbit = {
        "CNOT": qiskit_circuit.cx,
        "cX": qiskit_circuit.cx,
        "aCNOT": qiskit_circuit.cx,
        "acX": qiskit_circuit.cx,
        "cY": qiskit_circuit.cy,
        "cV": qiskit_circuit.csx,
        "SWAP": qiskit_circuit.swap
    }
    
    gate_map_1qbit_with_param = {
        "R": qiskit_circuit.r,
        "Rx": qiskit_circuit.rx,
        "Ry": qiskit_circuit.ry,
        "Rz": qiskit_circuit.rz,
        "rU1": qiskit_circuit.unitary
    }

    gate_map_2qubit_with_param = {
        "A": qiskit_circuit.unitary,
        "cR": qiskit_circuit.unitary,
        "cRz": qiskit_circuit.unitary,
        "rU2": qiskit_circuit.unitary
    }

    for gate in gates:
        if gate.target() == gate.control():
            if gate.gate_id() in gate_map_1qbit:
                gate_map_1qbit[gate.gate_id()](gate.target())
                # if gate.gate_id() == "Rzy": #special case for Rzy
                #     qiskit_circuit.unitary(Rzy_mat, [gate.target()], label=gate.gate_id())
                # elif gate.gate_id() == "adj(Rzy)":
                #     qiskit_circuit.unitary((Rzy_mat.conj().T), [gate.target()], label=gate.gate_id())
                # else:
                #     gate_map_1qbit[gate.gate_id()](gate.target())
            elif gate.gate_id() in gate_map_1qbit_with_param:
                if gate.gate_id() == "rU1": #special case for rU1
                    print("RU1 GATE!!!")
                    rU1_mat = np.array([[]]) #TODO
                    qiskit_circuit.unitary(rU1_mat, [gate.target()], label=gate.gate_id())
                else:
                    gate_map_1qbit_with_param[gate.gate_id()](np.real(gate.param()), gate.target())
            else:
                raise ValueError(f"Unsupported one-qubit gate: {gate.gate_id()}")
        else:
            if gate.gate_id() in gate_map_2qbit:
                gate_map_2qbit[gate.gate_id()](gate.control(), gate.target())
            elif gate.gate_id() in gate_map_2qubit_with_param:
                custom_mat = np.array([]) #TODO
                qiskit_circuit.unitary(custom_mat, [gate.control(), gate.target()], label=gate.gate_id())
            else:
                raise ValueError(f"Unsupported two-qubit gate: {gate.gate_id()}")
    
    return qiskit_circuit

def qforte_to_qiskit_V2(qforte_circuit: Circuit, nqubits: int) -> QuantumCircuit:
    #Initialize a QuantumCircuit
    qiskit_circuit = QuantumCircuit(nqubits)

    #Mappings between qForte gate IDs and Qiskit methods
    gate_map_1qbit = {
        "X": qiskit_circuit.x,
        "Y": qiskit_circuit.y,
        "Z": qiskit_circuit.z,
        "H": qiskit_circuit.h,
        "V": qiskit_circuit.sx,
        "S": qiskit_circuit.s,
        "T": qiskit_circuit.t,
        "I": qiskit_circuit.id,
    }

    gate_map_2qbit = {
        "CNOT": qiskit_circuit.cx,
        "cX": qiskit_circuit.cx,
        "aCNOT": lambda control, target: qiskit_circuit.cx(target, control), #aCNOT is a controlled X gate with control and target swapped
        "acX": lambda control, target: qiskit_circuit.cx(target, control),   #acX is the same as aCNOT
        "cY": qiskit_circuit.cy,
        "cV": qiskit_circuit.csx,
        "SWAP": qiskit_circuit.swap
    }
    
    gate_map_1qbit_with_param = {
        "R": qiskit_circuit.r,
        "Rx": qiskit_circuit.rx,
        "Ry": qiskit_circuit.ry,
        "Rz": qiskit_circuit.rz,
    }

    gate_set_1qbit_custom = {
        "Rzy",
        "adj(Rzy)",
        "rU1",
        "adj(rU1)"
    }
    gate_set_2qubit_custom = {
        "A",
        "adj(A)",
        "cR",
        "adj(cR)",
        "cRz",
        "adj(cRz)",
        "rU2",
        "adj(rU2)"
    }
    #Iterate through the gates in the qForte circuit
    #And apply the corresponding Qiskit methods
    for gate in qforte_circuit.gates():

        #1-qbit gate without parameter
        if gate.gate_id() in gate_map_1qbit:
            gate_map_1qbit[gate.gate_id()](gate.target())

        #2-qbit gate without parameter
        elif gate.gate_id() in gate_map_2qbit:
            gate_map_2qbit[gate.gate_id()](gate.control(), gate.target())

        #1-qbit gate with parameter (uses parameter extracted from gate.param())
        elif gate.gate_id() in gate_map_1qbit_with_param:
            gate_map_1qbit_with_param[gate.gate_id()](gate.param().real, gate.target())

        #Custom unitary gates (Need to test if this transpiles correctly)

        #1-qbit custom unitary gates
        elif gate.gate_id() in gate_set_1qbit_custom:
            qiskit_circuit.append(UnitaryGate(np.array(gate.gate(), dtype=complex), label=gate.gate_id()), [gate.target()])
        elif gate.gate_id() in gate_set_2qubit_custom:
            qiskit_circuit.append(UnitaryGate(np.array(gate.gate(), dtype=complex), label=gate.gate_id()), [gate.control(), gate.target()])
        else:
            raise ValueError(f"Unsupported gate: {gate.gate_id()}")
        
    #Return the constructed Qiskit circuit
    return qiskit_circuit

def qiskit_to_qforte():
    pass