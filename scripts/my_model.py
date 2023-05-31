# my_model.py
from mpi4py import MPI
import numpy as np
import pickle
from mpi4py.futures import MPIPoolExecutor

def my_model(data):
    # This could be any function that takes inputs and produces outputs
    x, y = data
    return x**2 + y**2

if __name__ == '__main__':
    # Define the input data
    data = np.arange(20).reshape(-1, 2)  # Let's say we have 10 sets of (x, y) pairs

    with MPIPoolExecutor() as executor:
        # Apply the model to the data
        results = list(executor.map(my_model, data))

    # Print out the results
    print(results)

    with open('py_test.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
