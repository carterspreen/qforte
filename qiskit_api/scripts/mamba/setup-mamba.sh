#!/bin/bash

#This script creates a conda env with deps to build QForte and run its Qiskit integration.
#Optionally, you may supply an argument, to override the default env name, which is "qforte-qiskit".

#EXIT ON ERROR OR ON REFERENCING AN UNDEFINED VARIABLE
set -eu || { echo "error: shell doesn't support basic POSIX features"; exit 1; }
#USE STRONGER PIPELINE ERROR CHECKING, IF AVAILABLE
(set -o pipefail) > /dev/null 2>&1 && set -o pipefail || echo "warning: continuing w/o pipefail"

#SET PYTHON VERSION
PYTHON_VERSION=3.13
#SELECT QFORTE CONDA DEPS
QFORTE_PACKAGES="openblas psi4 cmake pytest"
#SELECT QISKIT CONDA DEPS
QISKIT_PACKAGES="qiskit qiskit-ibm-runtime qiskit-aer qiskit-qasm3-import"

#INITIALIZE MICROMAMBA
micromamba --version > /dev/null 2>&1 && eval "$(micromamba shell hook -s bash)"

#SET MICROMAMBA ENV NAME
if [ $# -gt 0 ]; then
    SETUP_ENV=$1
else
    SETUP_ENV="qforte-qiskit"
fi

#DELETE ENV IF IT ALREADY EXISTS
if micromamba env list | awk '{print $1}' | grep -Fxq $SETUP_ENV; then
    micromamba env remove -n "$SETUP_ENV" -y
fi

#CREATE MICROMAMBA ENV
micromamba create -n "$SETUP_ENV" -y -c conda-forge python=$PYTHON_VERSION $QFORTE_PACKAGES $QISKIT_PACKAGES
