#!/bin/sh -l

cd /github/workspace

export SIGKAX_CUDA=yes
python3 setup.py build_ext --inplace
python3 -m pip install .
python3 -c 'import sigkax; print(sigkax.__version__)'
python3 - c 'from sigkax import cpu_ops'
python3 -c 'from sigkax import gpu_ops'