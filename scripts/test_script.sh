#!/bin/bash

#SBATCH --job-name="Python_test"
#SBATCH --time=00:00:10
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=education-tpm-msc-epa

module load 2022r2
module load openmpi
module load python
module load py-numpy
module load py-mpi4py

mpirun python my_model.py > py_test.log
