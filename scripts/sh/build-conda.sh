#!/bin/bash

# This is the development build script for qForte.
# It rebuilds and installs qForte.

# This script requires conda to be available in the PATH.

# This script accepts one (optional) positional parameter.
# It specifies the conda environment to build against.
# If not provided, will fall back to the currently active environment.
# If no environment is active, will fall back to "qforte-default-env".

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

#check for conda
which conda > /dev/null 2>&1 || { echo "conda not found..exiting"; exit 1;}

if [ $# -gt 0 ]; then
    QFORTE_CONDA_ENV=$1
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    QFORTE_CONDA_ENV="$CONDA_DEFAULT_ENV"
else
    QFORTE_CONDA_ENV="qforte-default-env"
fi

#ACTIVATE CONDA ENV
echo "Initializing Conda...\n"
source "$(conda info --base)/etc/profile.d/conda.sh"
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

#SET LIBOPENBLAS PATH DEPENDING ON OS
OS="$(uname)"
if [ "$OS" = "Linux" ]; then
	echo "You are on Linux: using libopenblas.so"
	sed -i 's|set(OPENBLAS_EXE ".*")|set(OPENBLAS_EXE ${CMAKE_PREFIX_PATH}/lib/libopenblas.so)|' CMakeLists.txt
elif [ "$OS" = "Darwin" ]; then
	echo "You are on MacOS: using libopenblas.dylib"
	sed -i 's|set(OPENBLAS_EXE ".*")|set(OPENBLAS_EXE ${CMAKE_PREFIX_PATH}/lib/libopenblas.dylib)|' CMakeLists.txt
	exit
else
	echo "Unknown OS -> exiting..."
	exit
fi

#BUILD
python setup.py develop && echo "qforte successfully installed" || { echo "qforte failed to install"; exit 1; }
