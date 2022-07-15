#!/bin/env python

import numpy as np
import pytest
import virgo.mpi.parallel_sort as psort

def assert_all_ranks(condition, message):
    """Fails assertion on all ranks if condition is False on any rank"""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    condition = comm.allreduce(condition, op=MPI.MIN)
    assert condition, message

def assert_rank_zero(condition, message):
    """
    Fails assertion on all ranks if condition is False on rank zero.
    Value of condition is not significant on other ranks.
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()    
    condition = comm.bcast(condition)
    assert condition, message

def run_parallel_sort(input_function, nr_tests):
    """
    Parallel sort nr_tests arrays generated by calling input_function to
    generate test data then calling virgo.mpi.parallel_sort.parallel_sort().

    Raises AssertionError if arrays are not sorted correctly or the sorting
    index cannot be used to reproduce the sorted array.
    """

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    for i in range(nr_tests):

        # Make the test array
        arr = input_function()

        # Parallel sort then gather result on rank 0
        arr_ps = arr.copy()
        index = psort.parallel_sort(arr_ps, return_index=True)
        arr_ps_g = psort.gather_to_first_rank(arr_ps, comm)

        # Gather on rank 0 then serial sort
        arr_g = psort.gather_to_first_rank(arr, comm)
        if comm_rank == 0:
            arr_g_ss = np.sort(arr_g)

        # Compare
        if comm_rank == 0:
            all_equal = np.all(arr_ps_g == arr_g_ss)
        else:
            all_equal = None
        assert_rank_zero(all_equal, "Array was not sorted correctly")

        # Check that we can reconstruct the sorted array using the index
        arr_ps_from_index = psort.fetch_elements(arr, index)
        assert_all_ranks(np.all(arr_ps_from_index == arr_ps), "Index does not reproduce sorted array!")

@pytest.mark.mpi
def test_parallel_sort_random_integers():

    def input_function():
        max_local_size = 10000
        max_value = 10
        n   = np.random.randint(max_local_size) + 0
        arr = np.random.randint(max_value, size=n)
        return arr

    run_parallel_sort(input_function, 200)

@pytest.mark.mpi
def test_parallel_sort_random_floats():

    def input_function():
        max_local_size = 10000
        max_value = 1.0e10
        n   = np.random.randint(max_local_size) + 0
        arr = np.random.uniform(low=-max_value, high=max_value, size=n)
        return arr

    run_parallel_sort(input_function, 200)

@pytest.mark.mpi
def test_parallel_sort_all_empty():

    def input_function():
        return np.zeros(0, dtype=float)
    run_parallel_sort(input_function, 1)

@pytest.mark.mpi
def test_parallel_sort_some_empty():

    def input_function():
        max_local_size = 10000
        max_value = 1.0e10
        n   = np.random.randint(max_local_size) + 0
        if np.random.randint(2) == 0:
            n = 0 # 50% chance for rank to have empty array
        arr = np.random.uniform(low=-max_value, high=max_value, size=n)
        return arr

    run_parallel_sort(input_function, 200)

@pytest.mark.mpi
def test_parallel_sort_unyt_floats():

    try:
        import unyt
    except ImportError:
        pytest.skip("Unable to import unyt")

    def input_function():
        max_local_size = 10000
        max_value = 1.0e10
        n   = np.random.randint(max_local_size) + 0
        arr = np.random.uniform(low=-max_value, high=max_value, size=n)
        return unyt.unyt_array(arr, units=unyt.cm)

    run_parallel_sort(input_function, 200)

@pytest.mark.mpi
def test_parallel_sort_structured_arrays():

    def input_function():
        max_local_size = 10000
        max_value = 10
        dtype = ([("a", int), ("b", int)])
        n   = np.random.randint(max_local_size) + 0
        arr = np.ndarray(n, dtype=dtype)
        arr["a"] = np.random.randint(max_value, size=n)
        arr["b"] = np.random.randint(max_value, size=n)
        return arr

    run_parallel_sort(input_function, 200)

@pytest.mark.mpi
def test_repartition_random_integers():

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    nr_tests = 200
    max_local_size = 1000
    max_value = 20

    for i in range(nr_tests):

        # Make random test aray
        n   = np.random.randint(max_local_size) + 0
        arr = np.random.randint(max_value, size=n)    

        # Pick random destination for each element
        dest   = np.random.randint(comm_size, size=n)

        # Count elements we want on each processor
        ndesired = np.zeros(comm_size, dtype=int)
        for rank in range(comm_size):
            nlocal = np.sum(dest==rank)
            ndesired[rank] = comm.allreduce(nlocal)

        # Repartition
        new_arr = psort.repartition(arr, ndesired)

        # Verify result by gathering on rank 0
        arr_gathered = np.concatenate(comm.allgather(arr))
        new_arr_gathered = np.concatenate(comm.allgather(new_arr))
        if comm_rank == 0:
            all_equal = np.all(arr_gathered == new_arr_gathered)
        else:
            all_equal = None
        assert_rank_zero(all_equal, "Repartitoned array is incorrect!")


@pytest.mark.mpi
def test_repartition_structured_array():

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    nr_tests = 200
    max_local_size = 1000
    max_value = 20

    dtype = np.dtype([("a", int), ("b", int)])

    for i in range(nr_tests):

        # Make random test aray
        n   = np.random.randint(max_local_size) + 0
        arr = np.ndarray(n, dtype=dtype)
        arr["a"] = np.random.randint(max_value, size=n)
        arr["b"] = np.random.randint(max_value, size=n)

        # Pick random destination for each element
        dest   = np.random.randint(comm_size, size=n)

        # Count elements we want on each processor
        ndesired = np.zeros(comm_size, dtype=int)
        for rank in range(comm_size):
            nlocal = np.sum(dest==rank)
            ndesired[rank] = comm.allreduce(nlocal)

        # Repartition
        new_arr = psort.repartition(arr, ndesired)

        # Verify result by gathering on rank 0
        arr_gathered = np.concatenate(comm.allgather(arr))
        new_arr_gathered = np.concatenate(comm.allgather(new_arr))
        if comm_rank == 0:
            all_equal = np.all(arr_gathered == new_arr_gathered)
        else:
            all_equal = None
        assert_rank_zero(all_equal, "Repartitoned array is incorrect!")


@pytest.mark.mpi
def test_unique():

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    nr_tests = 200
    max_size = 1000
    max_value = 50

    for test_nr in range(nr_tests):

        # Create test dataset
        nr_local_elements = np.random.randint(max_size)
        nr_total_elements = comm.allreduce(nr_local_elements)
        local_data = np.random.randint(max_value, size=nr_local_elements)

        # Find unique values
        local_unique, local_counts = psort.parallel_unique(local_data, comm=comm, return_counts=True,
                                                           repartition_output=True)

        # Combine values and counts on first rank
        global_data = comm.gather(local_data)
        global_unique = comm.gather(local_unique)
        global_counts = comm.gather(local_counts)
        if comm_rank == 0:
            global_data = np.concatenate(global_data)
            global_unique = np.concatenate(global_unique)
            global_counts = np.concatenate(global_counts)
            check_unique, check_counts = np.unique(global_data, return_counts=True)
            all_unique_equal = np.all(global_unique == check_unique)
            all_counts_equal = np.all(global_unique == check_unique)
        else:
            all_unique_equal = None
            all_counts_equal = None
        assert_rank_zero(all_unique_equal, "Unique values are incorrect")
        assert_rank_zero(all_counts_equal, "Unique counts are incorrect")

def run_large_parallel_sort(elements_per_rank):
    """
    Test parallel_sort() on larger input arrays.

    This sorts an array of small(ish) integers and checks that the number of
    instances of each value is preserved and that the result is in order.
    The number of elements on each rank is randomly chosen  between 0 and
    2*elements_per_rank.
    The array is never gathered one one rank, so larger test cases can be run.
    """
    
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Create input arrays
    max_value = 1000000
    n = np.random.randint(2*elements_per_rank)
    arr = np.random.randint(max_value, size=n)
    ntot = comm.allreduce(n)

    # Count number of instances of each value
    num_val_start = np.bincount(arr, minlength=max_value)
    num_val_start = comm.allreduce(num_val_start)

    # Sort the array
    arr_unsorted = arr.copy()
    index = psort.parallel_sort(arr, comm=comm, return_index=True)

    # Count number of instances of each value
    num_val_end = np.bincount(arr, minlength=max_value)
    num_val_end = comm.allreduce(num_val_end)
    
    # Check local ordering
    assert_all_ranks(np.all(arr[1:]>=arr[:-1]), "Array is not sorted correctly within a rank!")

    # Check ordering between ranks
    if comm_size > 1:
        rank_min_val = np.asarray(comm.allgather(np.amin(arr)))
        rank_max_val = np.asarray(comm.allgather(np.amax(arr)))
        assert_all_ranks(np.all(rank_min_val[1:] >= rank_max_val[:-1]), "Array is not sorted correctly between ranks!")

    # Check number of instances of each value has been preserved
    assert_all_ranks(np.all(num_val_start == num_val_end), "Sorted array is not a reordered copy of the input!")

    # Try to reconstruct the sorted array from the index
    arr_check = psort.fetch_elements(arr_unsorted, index, comm=comm)
    assert_all_ranks(np.all(arr_check==arr), "Index does not reproduce sorted array!")

@pytest.mark.mpi
def test_large_parallel_sort():
    run_large_parallel_sort(elements_per_rank=10000000)
    
