#!/bin/env python

import numpy as np


def collective_read(dataset):
    """
    Do a parallel collective read of a HDF5 dataset by splitting
    the dataset equally between MPI ranks along its first axis.
    
    File must have been opened in MPI mode.
    """

    # Avoid initializing HDF5 (and therefore MPI) until necessary
    import h5py

    # Find communicator file was opened with
    comm, info = dataset.file.id.get_access_plist().get_fapl_mpio()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Determine how many elements to read on each task
    ntot = dataset.shape[0]
    num_on_task = np.zeros(comm_size, dtype=int)
    num_on_task[:] = ntot // comm_size
    num_on_task[0:ntot % comm_size] += 1
    assert sum(num_on_task) == ntot

    # Determine offsets to read at on each task
    offset_on_task = np.cumsum(num_on_task) - num_on_task

    # Determine slice to read
    slice_to_read = np.s_[offset_on_task[comm_rank]:offset_on_task[comm_rank]+num_on_task[comm_rank],...]

    if ntot < 10*comm_size:
        # If the dataset is small, read on one rank and broadcast
        if comm_rank == 0:
            data = dataset[...]
        else:
            data = None
        data = comm.bcast(data)
        return data[slice_to_read]
    else:
        # Otherwise do a collective read
        with dataset.collective:
            data = dataset[slice_to_read]
    
    return data


def collective_write(group, name, data):
    """
    Do a parallel collective write of a HDF5 dataset by concatenating
    contributions from MPI ranks along the first axis.
    
    File must have been opened in MPI mode.
    """

    # Avoid initializing HDF5 (and therefore MPI) until necessary
    import h5py

    # Find communicator file was opened with
    comm, info = group.file.id.get_access_plist().get_fapl_mpio()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Determine how many elements to write on each task
    num_on_task = np.asarray(comm.allgather(data.shape[0]))
    ntot = np.sum(num_on_task)

    # Determine offsets at which to write data from each task
    offset_on_task = np.cumsum(num_on_task) - num_on_task

    # Need to have all dimensions but the first the same between ranks
    shape_local = tuple(data.shape[1:])
    shape_rank0 = tuple(comm.bcast(shape_local))
    if shape_local != shape_rank0:
        raise ValueError("Inconsistent data shapes in collective_write()!")

    # Need to have the same data type on all ranks
    dtype_local = data.dtype
    dtype_rank0 = comm.bcast(dtype_local)
    if dtype_local != dtype_rank0:
        raise ValueError("Inconsistent data types in collective_write()!")

    # Find the full shape of the new dataset
    full_shape = [ntot,] + list(shape_local)

    # Create the dataset
    dataset = group.create_dataset(name, shape=full_shape, dtype=dtype_rank0)

    # Determine slice to write
    slice_to_write = np.s_[offset_on_task[comm_rank]:offset_on_task[comm_rank]+num_on_task[comm_rank],...]

    # Do a collective write
    with dataset.collective:
        dataset[slice_to_write] = data

    return data
