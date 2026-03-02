#!/usr/bin/python

from qiskit_ibm_runtime import QiskitRuntimeService
 
try:
    QiskitRuntimeService()
    print("IBM Quantum Platform credentials found!")
except:
    print("IBM Quantum Platform credentials not found. Enter them in the input fields to get started: ")
    QiskitRuntimeService.save_account(
        token=input("Enter your IBM Quantum Platform API Key: "),
        name=input("Enter your IBMid account name: "),
        instance=input("Enter your IBM Quantum Platform instance name: "),
        set_as_default=True,
        overwrite=True,
)
