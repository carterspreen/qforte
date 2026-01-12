#!/bin/bash

# This is the first-time installer script for qForte
# It is tested and working on Arch, Debian, and Fedora.

# This script requires conda to be installed and available in the PATH.

# This script has the following effects:
#   1. Creates a conda environment with the needed packages
#   2. Edits the CMakeLists files (this is a temporary hack, those files need to be rewritten)
#   3. Builds and installs qForte inside the conda environment

# This script accepts one (optional) positional parameter:
#   It specifies the name of the conda environment that will be created.
#   If it is not passed, the name defaults to "qforte-default-env".

#EXIT ON ERROR
set -euo pipefail

#FIND THE PROJECT ROOT
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
  SOURCE="$(readlink -f "$SOURCE")"
done
SCRIPT_DIR="$(cd "$(dirname "$SOURCE")" && pwd)"

#RUN FROM PROJECT ROOT
#SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT" || exit 1

#ACCEPT ARGUMENTS
if [ $# -gt 0 ]; then
    QFORTE_CONDA_ENV=$1
else
    QFORTE_CONDA_ENV="qforte-qiskit-env"
fi

#CHECK FOR CONDA
which conda > /dev/null 2>&1 || { echo "conda not found..exiting"; exit 1;}

#INITIALIZE CONDA
echo "Initializing Conda..."
source "$(conda info --base)/etc/profile.d/conda.sh"

#CREATE CONDA ENV
echo "Creating Conda environment: $QFORTE_CONDA_ENV\n"
conda create -n "$QFORTE_CONDA_ENV" -y -c conda-forge python=3.10 openblas psi4 cmake pytest \
  qiskit qiskit-ibm-runtime qiskit-aer || exit 1

#ACTIVATE CONDA ENV
echo "Activating Conda environment..."
conda activate $QFORTE_CONDA_ENV

#SET THE CMAKE PREFIX TO THE CONDA PREFIX
echo "Setting CMAKE_PREFIX_PATH...\n"
sed -i "s|set(CMAKE_PREFIX_PATH \".*\")|set(CMAKE_PREFIX_PATH \"$CONDA_PREFIX\")|" CMakeLists.txt

#UPDATE THE MINIMUM REQUIRED VERSION OF CMAKE 
echo "Patching pybind11 and fmt CMakeLists.txt to use a supported version of CMake..."
#qforte
sed -i 's/^cmake_minimum_required(.*$/cmake_minimum_required(VERSION 3.5)/' CMakeLists.txt
#pybind11
sed -i 's/^cmake_minimum_required(.*$/cmake_minimum_required(VERSION 3.5)/' lib/pybind11/CMakeLists.txt
#fmt
sed -i 's/^cmake_minimum_required(.*$/cmake_minimum_required(VERSION 3.5)/' lib/fmt/CMakeLists.txt

#SET THE LIBOPENBLAS PATH DEPENDING ON OS
OS="$(uname)"
if [ "$OS" = "Linux" ]; then
	echo "You are on Linux: using libopenblas.so"
	sed -i 's|set(OPENBLAS_EXE ".*")|set(OPENBLAS_EXE ${CMAKE_PREFIX_PATH}/lib/libopenblas.so)|' CMakeLists.txt
elif [ "$OS" = "Darwin" ]; then
	echo "You are on MacOS: using libopenblas.dylib"
	sed -i 's|set(OPENBLAS_EXE ".*")|set(OPENBLAS_EXE ${CMAKE_PREFIX_PATH}/lib/libopenblas.dylib)|' CMakeLists.txt
else
	echo "OS not supported -> Exiting..."
	exit 1
fi

#BUILD
python setup.py develop && echo "qforte successfully installed" || { echo "qforte failed to install"; exit 1; }
