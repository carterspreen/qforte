import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from qforte.qiskit_api.translators import qforte_to_qiskit
from qforte.qiskit_api.dispatchers import QpuDispatcher, AerDispatcher, Dispatcher
from qiskit_ibm_runtime.fake_provider import FakeNighthawk
from qiskit.visualization import plot_histogram
from qiskit import qasm3

def writeQASM(circuit, f=None):

    qasm_str = qasm3.dumps(circuit)
    if f:
        try:
            fd = open(f, "w")
            fd.write(qasm_str)
        except Exception as e:
            raise RuntimeError(f"Failed to write QASM to file: {e}")

    return qasm_str

def readQASM(f):
    if f:
        try:
            circuit = qasm3.load(f)
            return circuit
        except Exception as e:
            raise RuntimeError(f"Failed to read QASM from file: {e}")

class SamplerHistFlow:

    def __init__(self, computer, circuit, shots=1024):
        self.circuit = circuit
        self.computer = computer
        self.shots = shots
        self.results = []

    def plot(self, legend=True):
        return self.hist()

    def get_or_cache(self, name, dispatcher, circuits, **kwargs):
        cache_file = f"{name}_result.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        result = dispatcher.dispatch_sampler(circuits=circuits, **kwargs)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        return result


    def hist(self):

        if not self.circuit:
        # Load the time evolution circuit from a QASM file
            qc = readQASM("data/circuit.qasm3")
            if not qc:
                raise ValueError("Failed to read the quantum circuit from the QASM file.")
        else:
            # make a copy so we don’t mutate the original (e.g. add measurements)
            qc = self.circuit.copy()

        qc.measure_all()

        # Create dispatchers for QPU and Aer
        qpu_dispatcher = QpuDispatcher()
        aer_dispatcher = AerDispatcher()
        fake_dispatcher = Dispatcher(FakeNighthawk())

        # Dispatch the circuits using QPU (cached)
        qpu_result = self.get_or_cache("qpu", qpu_dispatcher, circuits=[qc], shots=self.shots)

        # Dispatch the circuits using Aer (cached)
        aer_result = self.get_or_cache("aer", aer_dispatcher, circuits=[qc], shots=self.shots)
        fake_dispatcher_result = self.get_or_cache("fake", fake_dispatcher, circuits=[qc], shots=self.shots)

        # Extract data from the QPU results
        qpu_counts = qpu_result[0].data.meas.get_counts()
        aer_counts = aer_result[0].data.meas.get_counts()
        fake_brisbane_counts = fake_dispatcher_result[0].data.meas.get_counts()

        # plot the qpu results
        q_legend = ['IBM QPU', 'Simulated Nighthawk QPU', 'Aer Simulator']
        q_dists = [qpu_counts, fake_brisbane_counts, aer_counts]

        if self.computer:
            coeffs = self.computer.get_coeff_vec()
            probs = np.abs(coeffs) ** 2
            probs = [int(np.round(p) * self.shots) for p in probs]
            n = int(np.log2(len(probs)))
            bitstrings = [format(i, f'0{n}b') for i in range(len(probs))]
            amp_probs = dict(zip(bitstrings, probs))
            c_legend = ['QForte Simulator']
            c_dist = [amp_probs]
            legend = q_legend + c_legend
            dists = q_dists + c_dist
        else:
            legend = q_legend
            dists = q_dists


        ax = plot_histogram(dists,
                        title=r'Trotterized Time Evolution: $H_2 \to \hat{H} \to \prod_j\;  e^{-i H_j t}$',
                        legend=legend,
                        figsize=(10, 6),
                        bar_labels=False)
        plt.savefig("trotter_hist.png", dpi=300, bbox_inches='tight')
        plt.show()
        return dists