from mpi4py import MPI
import numpy as np
import pickle
from mpi4py.futures import MPIPoolExecutor

def my_model(data):
    x, y = data
    result = x**2 + y**2

    # Get the rank of the current MPI process.
    rank = MPI.COMM_WORLD.Get_rank()
    
    return rank, result

if __name__ == "__main__":
    data_list = [(i, j) for i in range(10) for j in range(10)]

    with MPIPoolExecutor() as executor:
        results = list(executor.map(my_model, data_list))

    # Print results along with the rank of the MPI process.
    for data, (rank, result) in zip(data_list, results):
        print(f"Rank: {rank}, Data: {data}, Result: {result}")

    with open('py_test.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
