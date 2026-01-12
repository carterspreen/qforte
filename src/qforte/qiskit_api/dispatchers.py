from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator
from qiskit.providers.backend import BackendV2
from qiskit_aer import AerSimulator
from qiskit import transpile

class Dispatcher:
    def __init__(self, backend=None):
        self.backend = backend if isinstance(backend, BackendV2) else None
        self.sampler = None
        self.estimator = None

    def get_backend(self):
        return self.backend

    def set_backend(self, backend):
        self.backend = backend if isinstance(backend, BackendV2) else None

    def dispatch_sampler(self, circuits, shots=None):
        if self.backend is None:
            raise ValueError("Backend must be set before dispatching sampler.")
        try:
            transpiled = transpile(circuits, backend=self.backend)
            if self.sampler is None:
                self.sampler = Sampler(mode=self.backend)
            job = self.sampler.run(transpiled, shots=shots)
        except Exception as e:
            raise RuntimeError(f"Failed to dispatch sampler: {e}")
        else:
            return job.result()

    def dispatch_estimator(self, circuits, observables, precision=None):
        if self.backend is None:
            raise ValueError("Backend must be set before dispatching estimator.")
        if len(observables != len(circuits)):
            raise ValueError("Number of observables must match number of circuits.")
        try:
            transpiled = transpile(circuits, backend=self.backend)
            pubs = zip(transpiled, observables)
            if self.estimator is None:
                self.estimator = Estimator(mode=self.backend)
            job = self.estimator.run(pubs, precision=precision)
        except Exception as e:
            raise RuntimeError(f"Failed to dispatch estimator: {e}")
        else:
            return job.result()

class QpuDispatcher(Dispatcher):

    def __init__(self, backend_name=None, service=None):
       self.service = QiskitRuntimeService() if not service else service
       super().__init__(backend=self.set_backend_from_name(backend_name))

    def set_backend_from_name(self, backend_name):
        try:
            return self.service.backend(backend_name) if backend_name else self.service.least_busy()
        except Exception as e:
            raise ValueError(f"Could not set backend: {e}")

class AerDispatcher(Dispatcher):

    def __init__(self):
        super().__init__(backend=AerSimulator())