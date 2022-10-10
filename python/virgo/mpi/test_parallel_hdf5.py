#!/bin/env python

import pytest
import numpy as np
import h5py
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort

def do_collective_read(tmp_path, max_local_size, chunk_size=None):
    """
    Write out a dataset in serial mode, read it back in collective mode
    then gather on rank zero to check that the contents are correct.
    """

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # If max_local_size is an int, make it a single element tuple
    try:
        max_local_size = tuple([int(i) for i in max_local_size])
    except TypeError:
        max_local_size = (int(max_local_size),)

    # Where to write the test file
    filepath = tmp_path / f"collective_read_test_{max_local_size}.hdf5"
    filepath = comm.bcast(filepath)

    # Generate test data on rank zero
    if comm_rank == 0:
        if max_local_size[0] > 0:
            n = np.random.randint(max_local_size[0]*comm_size)
        else:
            n = 0
        arr = np.random.uniform(low=-1.0e6, high=1.0e6, size=(n,)+max_local_size[1:])
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
def test_collective_read_empty(tmp_path):
    """
    Test collective read of an empty dataset
    """
    do_collective_read(tmp_path, 0)

@pytest.mark.mpi
def test_collective_read_1d(tmp_path):
    """
    Test collective reads of various size 1D datasets
    """
    for max_local_size in (1, 10, 100, 1000, 10000, 100000):
        do_collective_read(tmp_path, max_local_size)

@pytest.mark.mpi
def test_collective_read_small_chunks_1d(tmp_path):
    """
    Test collective reads of various size 1D datasets

    Uses small chunk size to test chunked reads
    """
    for max_local_size in (1, 10, 100, 1000, 10000, 100000):
        do_collective_read(tmp_path, max_local_size, chunk_size=256)

@pytest.mark.mpi
def test_collective_read_2d(tmp_path):
    """
    Test collective reads of various size 2D datasets
    """
    for max_local_size in (1, 10, 100, 1000, 10000, 100000):
        do_collective_read(tmp_path, (max_local_size, 3))

@pytest.mark.mpi
def test_collective_read_small_chunks_2d(tmp_path):
    """
    Test collective reads of various size 2D datasets

    Uses small chunk size to test chunked reads
    """
    for max_local_size in (1, 10, 100, 1000, 10000, 100000):
        do_collective_read(tmp_path, (max_local_size, 3), chunk_size=256)

def do_collective_write(tmp_path, max_local_size, chunk_size=None):
    """
    Write out a dataset in collective mode, read it back in serial mode
    then gather on rank zero to check that the contents are correct.
    """

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # If max_local_size is an int, make it a single element tuple
    try:
        max_local_size = tuple([int(i) for i in max_local_size])
    except TypeError:
        max_local_size = (int(max_local_size),)

    # Where to write the test file
    filepath = tmp_path / f"collective_read_test_{max_local_size}.hdf5"
    filepath = comm.bcast(filepath)

    # Generate the test data
    if max_local_size[0] > 0:
        n_local = np.random.randint(max_local_size[0])
    else:
        n_local = 0
    arr_local = np.random.uniform(low=-1.0e6, high=1.0e6, size=(n_local,)+max_local_size[1:])
    
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
def test_collective_write_empty(tmp_path):
    """
    Test collective write of an empty dataset
    """
    do_collective_write(tmp_path, 0)

@pytest.mark.mpi
def test_collective_write_1d(tmp_path):
    """
    Test collective writes of various size 1D datasets
    """
    for max_local_size in (1, 10, 100, 1000, 10000, 100000):
        do_collective_write(tmp_path, max_local_size)

@pytest.mark.mpi
def test_collective_write_small_chunks_1d(tmp_path):
    """
    Test collective writes of various size 1D datasets
    """
    for max_local_size in (1, 10, 100, 1000, 10000, 100000):
        do_collective_write(tmp_path, max_local_size, chunk_size=256)

@pytest.mark.mpi
def test_collective_write_2d(tmp_path):
    """
    Test collective writes of various size 2D datasets
    """
    for max_local_size in (1, 10, 100, 1000, 10000, 100000):
        do_collective_write(tmp_path, (max_local_size,3))

@pytest.mark.mpi
def test_collective_write_small_chunks_2d(tmp_path):
    """
    Test collective writes of various size 2D datasets
    """
    for max_local_size in (1, 10, 100, 1000, 10000, 100000):
        do_collective_write(tmp_path, (max_local_size,3), chunk_size=256)

def create_multi_file_output(tmp_path, basename, nr_files, elements_per_file,
                             group=None, have_missing=False):
    """
    Write an array distributed over multiple files
    """

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    
    if comm_rank == 0:

        # Decide which files will have no dataset
        if have_missing:
            assert nr_files > 1
            while True:
                have_data = np.random.rand(nr_files) < 0.5
                if sum(have_data) != 0 and sum(have_data) != nr_files:
                    break
        else:
            have_data = np.ones(nr_files, dtype=bool)

        # Loop over files to create
        for file_nr in range(nr_files):

            # Generate the filename
            file_path = tmp_path / f"{basename}.{file_nr}.hdf5"

            # Create the data for this file
            if elements_per_file > 0:
                n = np.random.randint(2*elements_per_file)
            else:
                n = 0
            arr = np.random.uniform(low=-1.0e6, high=1.0e6, size=n)

            # Write the file
            with h5py.File(file_path, "w") as outfile:
                if have_data[file_nr]:
                    if group is not None:
                        outfile.create_group(group)
                        outfile[group]["data"] = arr
                    else:
                        outfile["data"] = arr
                outfile["nr_files"] = nr_files
                h = outfile.create_group("Header")
                h.attrs["nr_files"] = nr_files

    comm.barrier()

def read_multi_file_output(tmp_path, basename, group=None):
    """
    Do a parallel read of a multi file output then gather the results on rank
    zero and check against a serial read.
    """

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Make format string for the file names
    filenames = str(tmp_path / f"{basename}.%(file_nr)d.hdf5")

    # Read the data using the MultiFile class
    mf = phdf5.MultiFile(filenames, file_nr_attr=("Header", "nr_files"), comm=comm)
    if group is None:
        arr = mf.read(("data",))["data"]
    else:
        arr = mf.read(("data",), group=group)["data"]
    
    # Gather array on rank zero
    arr = comm.allgather(arr)
    if comm_rank == 0:
        arr = np.concatenate(arr)

    # Read the data directly with h5py
    if comm_rank == 0:
        arr_check = []
        file_nr = 0
        while True:
            filename = filenames % {"file_nr":file_nr}
            try:
                with h5py.File(filename, "r") as infile:
                    if group is None:
                        if "data" in infile:
                            arr_check.append(infile["data"][...])
                    else:
                        if group in infile and "data" in infile[group]:
                            arr_check.append(infile[group]["data"][...])                        
            except FileNotFoundError:
                break
            else:
                file_nr += 1
        arr_check = np.concatenate(arr_check)

    # Compare
    if comm_rank == 0:
        all_equal = np.all(arr==arr_check)
    else:
        all_equal = None
    all_equal = comm.bcast(all_equal)
    assert all_equal, "Multi file output was not read correctly"

def do_multi_file_test(tmp_path, basename, nr_files, elements_per_file, group=None, have_missing=False):
    """Create and read in a set of files"""

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path) # Use same path on all ranks

    create_multi_file_output(tmp_path, basename, nr_files, elements_per_file, group=group,
                             have_missing=have_missing)
    read_multi_file_output(tmp_path, basename, group=group)

@pytest.mark.mpi
def test_multi_file_single_file(tmp_path):
    for n in (0, 1, 10, 100, 1000, 10000):
        do_multi_file_test(tmp_path, basename="single_file", nr_files=1, elements_per_file=n)

@pytest.mark.mpi
def test_multi_file_single_file_group(tmp_path):
    for n in (0, 1, 10, 100, 1000, 10000):
        do_multi_file_test(tmp_path, basename="single_file_group", nr_files=1, elements_per_file=n,
                           group="group")

@pytest.mark.mpi
def test_multi_file_more_files_than_ranks(tmp_path):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    for n in (0, 1, 10, 100, 1000, 10000):
        do_multi_file_test(tmp_path, basename="more_files", nr_files=comm_size+1, elements_per_file=n)

@pytest.mark.mpi
def test_multi_file_more_files_than_ranks_group(tmp_path):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    for n in (0, 1, 10, 100, 1000, 10000):
        do_multi_file_test(tmp_path, basename="more_files_group", nr_files=comm_size+1, elements_per_file=n,
                           group="group")

@pytest.mark.mpi
def test_multi_file_more_files_than_ranks_group_missing(tmp_path):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    for n in (0, 1, 10, 100, 1000, 10000):
        do_multi_file_test(tmp_path, basename="more_files_group_missing", nr_files=comm_size+1, elements_per_file=n,
                           group="group", have_missing=True)

@pytest.mark.mpi
def test_multi_file_more_ranks_than_files(tmp_path):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    nr_files = comm_size-1
    if nr_files <= 0:
        pytest.skip("Need >1 MPI rank for this test")

    for n in (0, 1, 10, 100, 1000, 10000):
        do_multi_file_test(tmp_path, basename="more_ranks", nr_files=nr_files, elements_per_file=n)

@pytest.mark.mpi
def test_multi_file_more_ranks_than_files_group(tmp_path):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    nr_files = comm_size-1
    if nr_files <= 0:
        pytest.skip("Need >1 MPI rank for this test")

    for n in (0, 1, 10, 100, 1000, 10000):
        do_multi_file_test(tmp_path, basename="more_ranks_group", nr_files=nr_files, elements_per_file=n,
                           group="group")

@pytest.mark.mpi
def test_multi_file_more_ranks_than_files_group_missing(tmp_path):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    nr_files = comm_size-1
    if nr_files <= 1:
        pytest.skip("Need >2 MPI ranks for this test")

    for n in (0, 1, 10, 100, 1000, 10000):
        do_multi_file_test(tmp_path, basename="more_ranks_group_missing", nr_files=nr_files, elements_per_file=n,
                           group="group", have_missing=True)

@pytest.mark.mpi
def test_multi_file_one_file_per_rank(tmp_path):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    for n in (0, 1, 10, 100, 1000, 10000):
        do_multi_file_test(tmp_path, basename="file_per_rank", nr_files=comm_size, elements_per_file=n)

@pytest.mark.mpi
def test_multi_file_one_file_per_rank_group(tmp_path):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    for n in (0, 1, 10, 100, 1000, 10000):
        do_multi_file_test(tmp_path, basename="file_per_rank_group", nr_files=comm_size, elements_per_file=n,
                           group="group")

def multi_file_round_trip(tmp_path, nr_files, elements_per_file, basename, have_missing=False, group=None):
    """
    Check that writing out a distributed array to a file set
    then reading it back in preserves the values.
    """

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    # Sync path between ranks
    tmp_path = comm.bcast(tmp_path)

    # Create the data to read/write
    create_multi_file_output(tmp_path, basename, nr_files, elements_per_file, have_missing=have_missing, group=group)
    comm.barrier()

    # Make format string for the file names
    filenames1 = str(tmp_path / f"{basename}.%(file_nr)d.hdf5")

    # Read the data using the MultiFile class
    mf = phdf5.MultiFile(filenames1, file_nr_attr=("Header", "nr_files"), comm=comm)
    data = mf.read(("data",), group=group)["data"]

    # Write the same data to a new set of files
    elements_per_file = mf.get_elements_per_file("data", group=group)
    filenames2 = str(tmp_path / f"{basename}.mf_write_test.%(file_nr)d.hdf5")
    mf.write({"data" : data}, elements_per_file, filenames2, "w", group=group)

    comm.barrier()

    # Check that the file sets have the same contents
    if comm.Get_rank() == 0:
        equal = True
        for file_nr in range(nr_files):
            filename1 = filenames1 % {"file_nr" : file_nr}
            with h5py.File(filename1, "r") as infile:
                # Find group to read from (it may not exist)
                if group is None:
                    loc = infile
                elif group in infile:
                    loc = infile[group]
                else:
                    loc = None
                # Read dataset if possible
                if loc is not None and "data" in loc:
                    arr1 = loc["data"][...]
                else:
                    arr1 = ()
            # MultiFile.write() should always create a dataset, even if zero sized
            filename2 = filenames2 % {"file_nr" : file_nr}
            with h5py.File(filename2, "r") as infile:
                loc = infile if group is None else infile[group]
                if "data" in loc:
                    arr2 = loc["data"][...]
                else:
                    arr2 = ()
            # The arrays should be identical
            equal = equal & (len(arr1)==len(arr2)) & np.all(arr1==arr2)
    else:
        equal = None
    equal = comm.bcast(equal)
    assert equal, "Array written to file set did not round trip"

@pytest.mark.mpi
def test_round_trip_single_file(tmp_path):
    multi_file_round_trip(tmp_path, nr_files=1, elements_per_file=10000,
                          basename="round_trip_single_file")

@pytest.mark.mpi
def test_round_trip_file_per_rank(tmp_path):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    multi_file_round_trip(tmp_path, nr_files=comm_size, elements_per_file=10000,
                          basename="round_trip_file_per_rank")

@pytest.mark.mpi
def test_round_trip_few_files(tmp_path):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    nr_files = max(1, (comm_size // 2))
    multi_file_round_trip(tmp_path, nr_files=nr_files, elements_per_file=10000,
                          basename="round_trip_few_files")

@pytest.mark.mpi
def test_round_trip_few_files_missing(tmp_path):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    nr_files = max(1, (comm_size // 2))
    if nr_files <= 1:
        pytest.skip("Need >3 MPI ranks for this test")

    multi_file_round_trip(tmp_path, nr_files=nr_files, elements_per_file=10000,
                          basename="round_trip_few_files_missing", have_missing=True)

@pytest.mark.mpi
def test_round_trip_few_files_missing_group(tmp_path):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    nr_files = max(1, (comm_size // 2))
    if nr_files <= 1:
        pytest.skip("Need >3 MPI ranks for this test")

    multi_file_round_trip(tmp_path, nr_files=nr_files, elements_per_file=10000,
                          basename="round_trip_few_files_missing_group",
                          have_missing=True, group="group")

@pytest.mark.mpi
def test_round_trip_many_files(tmp_path):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    nr_files = comm_size * 2
    multi_file_round_trip(tmp_path, nr_files=nr_files, elements_per_file=10000,
                          basename="round_trip_many_files")

@pytest.mark.mpi
def test_round_trip_many_files_missing(tmp_path):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    nr_files = comm_size * 2
    multi_file_round_trip(tmp_path, nr_files=nr_files, elements_per_file=10000,
                          basename="round_trip_many_files_missing", have_missing=True)

@pytest.mark.mpi
def test_round_trip_many_files_missing_group(tmp_path):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    nr_files = comm_size * 2
    multi_file_round_trip(tmp_path, nr_files=nr_files, elements_per_file=10000,
                          basename="round_trip_many_files_missing_group",
                          have_missing=True, group="group")

