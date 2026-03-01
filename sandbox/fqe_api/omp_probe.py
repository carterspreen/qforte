# omp_probe.py
import os, ctypes
print("env OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))

lib = ctypes.CDLL("libomp.so")
lib.omp_get_max_threads.restype = ctypes.c_int
print("initial omp_get_max_threads:", lib.omp_get_max_threads())

import numpy as np
print("after numpy omp_get_max_threads:", lib.omp_get_max_threads())

import fqe
print("after fqe omp_get_max_threads:", lib.omp_get_max_threads())

import qforte
print("after qforte omp_get_max_threads:", lib.omp_get_max_threads())
