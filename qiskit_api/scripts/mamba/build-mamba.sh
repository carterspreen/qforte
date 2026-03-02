#!/bin/bash

#This script builds and installs QForte within a micromamba environment.

#Optionally, you may supply an argument, to specify the mamba environment you wish to link.
#If no argument is provided, the script will use the currently active environment.
#If no environment is active, the script will fall back to "qforte-qiskit".

#EXIT ON ERROR OR ON REFERENCING AN UNDEFINED VARIABLE
set -eu || { echo "shell doesn't support basic POSIX features"; exit 1; }
#USE STRONGER PIPELINE ERROR CHECKING, IF AVAILABLE
(set -o pipefail) > /dev/null 2>&1 && set -o pipefail || echo "continuing w/o pipefail"

#RUN FROM THE PROJECT ROOT
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
  SOURCE="$(readlink -f "$SOURCE")"
done
CDPATH= cd "$(dirname "$SOURCE")/../../.." || { echo "project root not found"; exit 1; }

#CHECK FOR MICROMAMBA
micromamba --version > /dev/null 2>&1 || { echo "micromamba not found"; exit 1;}

#INITIALIZE MICROMAMBA
eval "$(micromamba shell hook -s bash)"

#SELECT MICROMAMBA ENV
if [ $# -gt 0 ]; then
    BUILD_ENV=$1
elif [[ -v CONDA_DEFAULT_ENV ]]; then
    BUILD_ENV="$CONDA_DEFAULT_ENV"
else
    BUILD_ENV="qforte-qiskit"
fi
echo "using env: $BUILD_ENV"

#ACTIVATE MICROMAMBA ENV
micromamba activate $BUILD_ENV || { echo "micromamba env not found"; exit 1; }

#BACKUP CMakeLists.txt
cp -f CMakeLists.txt CMakeLists.txt.tmp || { echo "failed to backup CMakeLists.txt"; exit 1; }

#SET THE CMAKE PREFIX CORRECTLY
sed -i "s|set(CMAKE_PREFIX_PATH \".*\")|set(CMAKE_PREFIX_PATH \"$CONDA_PREFIX\")|" CMakeLists.txt

#BUILD QFORTE AND SAVE THE STATUS CODE
if python setup.py develop; then
    BUILD_EXIT=0; echo "build succeeded"
else
    BUILD_EXIT=1; echo "build failed"
fi

#RESTORE CMakeLists.txt
mv -f CMakeLists.txt.tmp CMakeLists.txt || { echo "failed to restore CMakeLists.txt"; exit 1; }

#EXIT WITH THE STATUS CODE OF THE BUILD
exit $BUILD_EXIT
