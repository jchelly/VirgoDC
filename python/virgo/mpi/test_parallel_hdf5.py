#!/bin/env python

import pytest
import numpy as np
import h5py
import virgo.mpi.parallel_hdf5 as phdf5

def do_collective_read(tmp_path, max_local_size, chunk_size=None):
    """
    Write out a dataset in serial mode, read it back in collective mode
    then gather on rank zero to check that the contents are correct.
    """

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Where to write the test file
    filepath = tmp_path / f"collective_read_test_{max_local_size}.hdf5"
    filepath = comm.bcast(filepath)

    # Generate test data on rank zero
    if comm_rank == 0:
        n   = np.random.randint(max_local_size*comm_size)
        arr = np.random.uniform(low=-1.0e6, high=1.0e6, size=n)
        with h5py.File(filepath, "w") as outfile:
            outfile["data"] = arr
    comm.barrier()

    # Read back in the test data
    with h5py.File(filepath, "r", driver="mpio", comm=comm) as infile:
        arr_coll = phdf5.collective_read(infile["data"], comm, chunk_size)

    # Check the result on rank 0
    arr_coll = comm.gather(arr_coll)
    if comm_rank == 0:
        arr_coll = np.concatenate(arr_coll)
        all_equal = np.all(arr==arr_coll)
    else:
        all_equal = None
    all_equal = comm.bcast(all_equal)
    assert all_equal, "Collective read returned incorrect data"

@pytest.mark.mpi
def test_collective_read(tmp_path):
    """
    Test collective reads of various size datasets
    """
    for max_local_size in (1, 10, 100, 1000, 10000, 100000):
        do_collective_read(tmp_path, max_local_size)

@pytest.mark.mpi
def test_collective_read_small_chunks(tmp_path):
    """
    Test collective reads of various size datasets

    Uses small chunk size to test chunked reads
    """
    for max_local_size in (1, 10, 100, 1000, 10000, 100000):
        do_collective_read(tmp_path, max_local_size, chunk_size=256)

def do_collective_write(tmp_path, max_local_size, chunk_size=None):
    """
    Write out a dataset in collective mode, read it back in serial mode
    then gather on rank zero to check that the contents are correct.
    """

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Where to write the test file
    filepath = tmp_path / f"collective_read_test_{max_local_size}.hdf5"
    filepath = comm.bcast(filepath)

    # Generate the test data
    n_local   = np.random.randint(max_local_size)
    arr_local = np.random.uniform(low=-1.0e6, high=1.0e6, size=n_local)
    
    # Write out the data in collective mode
    with h5py.File(filepath, "w", driver="mpio", comm=comm) as outfile:
        phdf5.collective_write(outfile, "data", arr_local, comm, chunk_size)
    comm.barrier()

    # Gather data on rank zero and check
    arr = comm.gather(arr_local)
    if comm_rank == 0:
        arr = np.concatenate(arr)
        with h5py.File(filepath, "r") as infile:
            arr_check = infile["data"][...]
        all_equal = np.all(arr==arr_check)
    else:
        all_equal = None
    all_equal = comm.bcast(all_equal)

    assert all_equal, "Collective write returned incorrect data"

@pytest.mark.mpi
def test_collective_write(tmp_path):
    """
    Test collective writes of various size datasets
    """
    for max_local_size in (1, 10, 100, 1000, 10000, 100000):
        do_collective_write(tmp_path, max_local_size)

@pytest.mark.mpi
def test_collective_write_small_chunks(tmp_path):
    """
    Test collective writes of various size datasets
    """
    for max_local_size in (1, 10, 100, 1000, 10000, 100000):
        do_collective_write(tmp_path, max_local_size, chunk_size=256)
