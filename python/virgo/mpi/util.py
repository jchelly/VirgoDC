#!/bin/env python

import numpy as np
import virgo.mpi.gather_array as ga
import virgo.mpi.parallel_sort as ps

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

    This can be used for interpreting subfind subhalo_tab/ids files
    or VELOCIraptor output.
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

    # Compute index of each group stored locally
    nr_groups_local = len(length)
    index_offset = comm.scan(nr_groups_local) - nr_groups_local
    index = np.arange(nr_groups_local, dtype=np.int64) + index_offset

    # Find range of particle IDs stored on each rank
    first_id_offset_local = comm.scan(nr_local_ids) - nr_local_ids
    first_id_offset = comm.allgather(first_id_offset_local)
    last_id_offset_local  = comm.scan(nr_local_ids) - 1
    last_id_offset = comm.allgather(last_id_offset_local)
    
    # Find the range of ranks we need to send each group's length, offset and index
    rank_send_offset = -np.ones(comm_size, dtype=int)
    rank_send_count = np.zeros(comm_size, dtype=int)
    first_rank_to_send_group_to = 0
    last_rank_to_send_group_to = -1
    for i in range(nr_groups_local):
        # Find first rank this group should be sent to
        while first_rank_to_send_group_to < comm_size-1 and last_id_offset[first_rank_to_send_group_to] < offset[i]:
            first_rank_to_send_group_to += 1
        # Find last rank this group should be sent to
        while last_rank_to_send_group_to < comm_size-1 and first_id_offset[last_rank_to_send_group_to+1] < offset[i]+length[i]:
            last_rank_to_send_group_to += 1
        # Accumulate number of groups to send to each rank
        for dest in range(first_rank_to_send_group_to, last_rank_to_send_group_to+1):
            if rank_send_offset[dest] < 0:
                rank_send_offset[dest] = i
            rank_send_count[dest] += 1

    # Find number of groups to receive on each rank and offset into receive buffers
    rank_recv_count = np.empty_like(rank_send_count)
    comm.Alltoall(rank_send_count, rank_recv_count)
    rank_recv_offset = np.cumsum(rank_recv_count) - rank_recv_count

    # Construct receive buffers
    nr_recv = np.sum(rank_recv_count)
    length_recv = np.ndarray(nr_recv, dtype=length.dtype)
    offset_recv = np.ndarray(nr_recv, dtype=offset.dtype)
    index_recv  = np.ndarray(nr_recv, dtype=index.dtype)

    # Exchange group lengths, offsets and indexes
    ps.my_alltoallv(length,      rank_send_count, rank_send_offset,
                    length_recv, rank_recv_count, rank_recv_offset,
                    comm=comm)
    ps.my_alltoallv(offset,      rank_send_count, rank_send_offset,
                    offset_recv, rank_recv_count, rank_recv_offset,
                    comm=comm)
    ps.my_alltoallv(index,       rank_send_count, rank_send_offset,
                    index_recv,  rank_recv_count, rank_recv_offset,
                    comm=comm)

    # Assign group membership to local particle IDs
    nr_ids_prev = comm.scan(nr_local_ids) - nr_local_ids
    grnr = -np.ones(nr_local_ids, dtype=np.int32)
    i1 = offset_recv - nr_ids_prev
    i2 = offset_recv + length_recv - nr_ids_prev
    i1[i1 < 0] = 0
    i2[i2 > nr_local_ids] = nr_local_ids
    for ind, start, end in zip(index_recv, i1, i2):
        if end > start:
            grnr[start:end] = ind
            
    return grnr
