#!/bin/bash

#SBATCH --job-name="Python_test"
#SBATCH --time=00:03:00
#SBATCH --ntasks=2
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

## Create directories on the scratch storage and link to them in your home directory.
mkdir -p /scratch/${USER}/.local
ln -s /scratch/${USER}/.local $HOME/.local

pip install --user -U ema_workbench

python ema_model.py > ema_test.log

# mpiexec -n 10 python -m mpi4py.futures my_model.py > py_test.log
