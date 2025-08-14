#!/bin/bash

#SBATCH --job-name="Python_test"
#SBATCH --time=00:06:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=3968mb
#SBATCH --account=research-tpm-mas

module load 2024r1
module load openmpi
module load python

mpiexec -n 1 python3 example_mpi_lake_model.py

