#!/bin/bash

#SBATCH --job-name="Python_test"
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=research-tpm-mas

module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-scipy
module load py-mpi4py
module load py-pip

pip install --user -e git+https://github.com/quaquel/EMAworkbench@mpi_update#egg=ema-workbench

mpiexec -n 1 python3 example_mpi_lake_model.py
