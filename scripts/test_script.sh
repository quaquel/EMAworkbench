#!/bin/bash

#SBATCH --job-name="Python_test"
#SBATCH --time=00:02:00
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-tpm-mas

module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-mpi4py
module load py-pip

python -m pip install --user -U -e git+https://github.com/quaquel/EMAworkbench@multi-node-development#egg=ema-workbench

mpiexec -n 100 python -m mpi4py.futures  ema_model.py > ema_test.log
