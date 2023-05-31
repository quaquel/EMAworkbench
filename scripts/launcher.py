# launcher.py
from mpi4py.futures import MPIPoolExecutor
import my_model

if __name__ == '__main__':
    with MPIPoolExecutor() as executor:
        executor.submit(my_model)
