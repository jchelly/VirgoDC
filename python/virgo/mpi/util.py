#!/bin/env python

import numpy as np
import virgo.mpi.gather_array as ga

def broadcast_dtype_and_dims(arr, comm=None):
    """
    Determine dtype of the specified array, arr, which may be 
    None on ranks which have no local elements.

    Also returns shape[1:] for the array.

    Will return (None, None) if array is None on all tasks.
    """

    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    if arr is not None:
        arr_dtype = arr.dtype
        arr_shape = arr.shape[1:]
    else:
        arr_dtype = None
        arr_shape = None

    arr_dtypes = comm.allgather(arr_dtype)
    for adt in arr_dtypes:
        if adt is not None and arr_dtype is None:
            arr_dtype = adt

    arr_shapes = comm.allgather(arr_shape)
    for ashp in arr_shapes:
        if ashp is not None and arr_shape is None:
            arr_shape = ashp

    return arr_dtype, arr_shape


def replace_none_with_zero_size(arr, comm=None):
    """
    Given an array which may be None on some tasks,
    return the array itself on tasks where it is not
    None, or a zero element array with appropriate
    type and dimensions on tasks where it is None.

    This can be useful for reading Gadget snapshots
    in parallel because Gadget omits datasets that
    would have zero size so some tasks might not know
    the type or dimensionality of the arrays in the 
    snapshot unless we do some communication.

    Will return None if arr is None on all tasks.
    """

    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    
    arr_dtype, arr_shape = broadcast_dtype_and_dims(arr, comm)

    if arr_dtype is None:
        return None
    elif arr is None:
        return np.ndarray([0,]+list(arr_shape), dtype=arr_dtype)
    else:
        return arr


def group_index_from_length_and_offset(length, offset, nr_local_ids, comm=None):
    """
    Given distributed arrays with the lengths and offsets
    of groups in an array of particle IDs, compute the group
    index corresponding to each particle ID.

    length: array with the lengths of the groups
    offset: array with the offsets to the groups
    nr_local_ids: size of the local part of the particle ID array

    This can be used for interpreting subfind subhalo_tab/ids files.
    """

    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Ensure lengths and offsets are signed, 64 bit ints -
    # prevents numpy casting to float when mixing signed and unsigned.
    length = np.asarray(length, dtype=np.int64)
    offset = np.asarray(offset, dtype=np.int64)

    # Gather the lengths and offsets
    all_lengths = ga.allgather_array(length, comm=comm)
    all_offsets = ga.allgather_array(offset, comm=comm)

    # Find number of IDs on lower numbered ranks
    nr_ids_prev = comm.scan(nr_local_ids) - nr_local_ids

    # Make array with group membership for each local particle
    grnr = -np.ones(nr_local_ids, dtype=np.int32)
    i1 = all_offsets - nr_ids_prev
    i2 = all_offsets + all_lengths - nr_ids_prev
    i1[i1 < 0] = 0
    i2[i2 > nr_local_ids] = nr_local_ids
    ind = i2 > i1    
    for i, (start, end) in enumerate(zip(i1[ind], i2[ind])):
        grnr[start:end] = i
        
    return grnr
