import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qforte.qiskit_api.dispatchers import QpuDispatcher
from qforte.qiskit_api.dispatchers import AerDispatcher
from qiskit import QuantumCircuit

def main():

    # Create a Bell state circuit
    qc_bell = QuantumCircuit(2, 2)
    qc_bell.h(0)
    qc_bell.cx(0, 1)
    qc_bell.measure_all()

    # Create dispatchers for QPU and Aer
    qpu_dispatcher = QpuDispatcher()
    aer_dispatcher = AerDispatcher()

    #dispatch the circuits
    aer_result_bell = aer_dispatcher.dispatch_sampler(circuits=[qc_bell], shots=1024)
    qpu_result_bell = qpu_dispatcher.dispatch_sampler(circuits=[qc_bell], shots=1024)

    #extract data from the results
    qpu_counts_bell = qpu_result_bell[0].data.meas.get_counts()
    aer_counts_bell = aer_result_bell[0].data.meas.get_counts()

    #plot the results
    plot_histogram(aer_counts_bell, title="Aer Bell State Circuit Counts")
    plot_histogram(qpu_counts_bell, title="Bell State Circuit Counts")
    plt.show()

if __name__ == "__main__":
    main()
