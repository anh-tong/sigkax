#!/bin/sh -l

cd /github/workspace

SIGKAX_CUDA=yes python3 -m pip install .
python3 -c 'import sigkax; print(sigkax.__version__)'
python3 - c 'from sigkax import cpu_ops'
python3 -c 'from sigkax gpu_ops'