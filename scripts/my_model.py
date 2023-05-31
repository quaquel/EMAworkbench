# my_model.py
from mpi4py import MPI
import numpy as np
import pickle

def my_model(x, y):
    # This could be any function that takes inputs and produces outputs
    return x**2 + y**2

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the input data
data = np.arange(20).reshape(-1, 2)  # Let's say we have 10 sets of (x, y) pairs

# Divide the data among the available processes
chunks = np.array_split(data, size)

# Each process gets its chunk of the data
chunk = chunks[rank]

# Each process applies the model to its chunk of the data
results = np.array([my_model(x, y) for x, y in chunk])

# Gather the results back to the root process
all_results = comm.gather(results, root=0)

if rank == 0:
    # Root process prints out the results
    print(np.concatenate(all_results))

    with open('py_test.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

