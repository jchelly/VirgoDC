#!/bin/env python

from __future__ import print_function

import numpy as np

import time
import sys
import gc

# Type to use for global indexes
index_dtype = np.int64

# Default algorithm for alltoallv function
default_alltoallv_method = "hypercube"


def mpi_datatype(dtype):
    """
    Return an MPI datatype corresponding to the supplied numpy type.
    """
    
    # Deal with types which have explicit big or little endian byte order
    # but are still native endian really
    if ((sys.byteorder == 'little' and dtype.byteorder == '<') or
        (sys.byteorder == 'big' and dtype.byteorder == '>')):
        dtype = dtype.newbyteorder("=")
    
    # Create the new MPI type
    import mpi4py.util.dtlib
    mpi_type = mpi4py.util.dtlib.from_numpy_dtype(dtype)
    mpi_type.Commit()

    return mpi_type


def my_alltoallv(sendbuf, send_count, send_offset,
                 recvbuf, recv_count, recv_offset,
                 comm=None, method=None):
    """
    Alltoallv implemented using sendrecv calls. Avoids problems
    caused when some ranks send or receive more than 2^31
    elements by splitting communications.
    """

    # Determine method to use
    if method is None:
        method = default_alltoallv_method
    
    # Maximum number of elements per message: avoid messages > 2GB
    assert sendbuf.dtype == recvbuf.dtype
    nchunk = (100*1024*1024) // sendbuf.dtype.itemsize

    # Get communicator to use
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    ptask = 0
    while(2**ptask < comm_size):
        ptask += 1

    # Find MPI data types
    mpi_type_send = mpi_datatype(sendbuf.dtype)
    mpi_type_recv = mpi_datatype(recvbuf.dtype)

    if method == "hypercube":
        #
        # Loop over pairs of processes and sendrecv() data
        #
        for ngrp in range(2**ptask):
            rank = comm_rank ^ ngrp
            if rank < comm_size:
                # Offsets to next block to send
                current_send_offset = send_offset[rank]
                left_to_send        = send_count[rank]
                current_recv_offset = recv_offset[rank]
                left_to_recv        = recv_count[rank]
                # Loop until all data has been sent
                while left_to_send > 0 or left_to_recv > 0:
                    # Find number to send this time
                    num_to_send = min((left_to_send, nchunk))
                    num_to_recv = min((left_to_recv, nchunk))
                    # Transfer the data
                    send_buf_spec = [sendbuf[current_send_offset:current_send_offset+num_to_send], mpi_type_send]
                    recv_buf_spec = [recvbuf[current_recv_offset:current_recv_offset+num_to_recv], mpi_type_recv]
                    comm.Sendrecv(send_buf_spec, rank, 0, recv_buf_spec, rank, 0)
                    # Update counts and offsets
                    left_to_send        -= num_to_send
                    current_send_offset += num_to_send
                    left_to_recv        -= num_to_recv
                    current_recv_offset += num_to_recv
    elif method == "async":
        #
        # Post non-blocking receives
        #
        receives = []
        for rank in range(comm_size):
            current_recv_offset = recv_offset[rank]
            left_to_recv        = recv_count[rank]
            while left_to_recv > 0:
                num_to_recv = min((left_to_recv, nchunk))
                recv_buf_spec = [recvbuf[current_recv_offset:current_recv_offset+num_to_recv], mpi_type_recv]
                receives.append(comm.Irecv(recv_buf_spec, rank))
                left_to_recv        -= num_to_recv
                current_recv_offset += num_to_recv
        #
        # Post non-blocking sends
        #
        sends = []
        for rank in range(comm_size):
            current_send_offset = send_offset[rank]
            left_to_send        = send_count[rank]
            while left_to_send > 0:
                num_to_send = min((left_to_send, nchunk))
                send_buf_spec = [sendbuf[current_send_offset:current_send_offset+num_to_send], mpi_type_send]
                sends.append(comm.Isend(send_buf_spec, rank))
                left_to_send        -= num_to_send
                current_send_offset += num_to_send
        #
        # Wait for everything to complete
        #
        MPI.Request.Waitall(sends+receives)
    else:
        raise ValueError("Unrecognised value of method parameter for alltoall")
        
    mpi_type_send.Free()
    mpi_type_recv.Free()


def my_argsort(arr):
    """
    Determines serial sorting algorithm used.
    """
    # Use mergesort because
    # - it's stable (affects behaviour of parallel_match if duplicate values exist)
    # - it's worst case performance is reasonable
    return np.argsort(arr, kind='mergesort')


def sendrecv(dest, sendbuf, recvbuf, comm=None, nchunk=None):
    """
    Sendrecv implementation which splits communications where necessary.
    Source and destination are assumed to be the same. Arrays must be 1D.
    
    dest    - other MPI rank to communicate with
    sendbuf - array to send
    recvbuf - array to receive into
    comm    - communicator to use, defaults to MPI_COMM_WORLD
    nchunk  - maximum number of elements per send (default 100MB)
    """

    # Get communicator to use
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    # Get data types
    mpi_type_send = mpi_datatype(sendbuf.dtype)
    mpi_type_recv = mpi_datatype(recvbuf.dtype)
    
    # Maximum number of elements per message: avoid messages > 2GB
    assert sendbuf.dtype == recvbuf.dtype
    assert len(sendbuf.shape) == 1
    assert len(recvbuf.shape) == 1
    if nchunk is None:
        nchunk = (100*1024*1024) // sendbuf.dtype.itemsize
    
    # Determine number of elements to send and receive
    nr_send_left = sendbuf.shape[0]
    nr_recv_left = comm.sendrecv(nr_send_left, dest=dest, source=dest)

    # Transfer data until it has all been moved
    send_offset = 0
    recv_offset = 0
    while (nr_send_left > 0) or (nr_recv_left > 0):
        nr_send = min(nchunk, nr_send_left)
        nr_recv = min(nchunk, nr_recv_left)
        comm.Sendrecv(sendbuf[send_offset:send_offset+nr_send], dest,
                      recvbuf=recvbuf[recv_offset:recv_offset+nr_recv],
                      source=dest)
        send_offset  += nr_send
        nr_send_left -= nr_send
        recv_offset  += nr_recv
        nr_recv_left -= nr_recv

    mpi_type_send.Free()
    mpi_type_recv.Free()


def alltoall_exchange(sendbuf, send_count, comm=None, reverse=False):
    """
    Carry out an alltoallv assuming contiguous array sections to be sent
    to each rank so that all counts and offsets can be computed from
    just the send counts.
    """

    # Get communicator to use
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    # Construct counts and offsets
    send_count  = np.asarray(send_count, dtype=int)
    send_offset = np.cumsum(send_count) - send_count
    recv_count  = np.ndarray(len(send_count), dtype=int)
    comm.Alltoall(send_count, recv_count)
    recv_offset = np.cumsum(recv_count) - recv_count

    if reverse:
        # Swap the send and receive counts and offsets in the case where they
        # describe a previous exchange which we now want to reverse.
        send_count, recv_count = recv_count, send_count
        send_offset, recv_offset = recv_offset, send_offset

    # Construct the output buffer
    nr_recv = recv_count.sum()
    recvbuf = np.ndarray(nr_recv, dtype=sendbuf.dtype)

    # Exchange data
    my_alltoallv(sendbuf, send_count, send_offset,
                 recvbuf, recv_count, recv_offset,
                 comm=comm)
    
    return recvbuf

        
def repartition(arr, ndesired, comm=None):
    """Return the input arr repartitioned between processors"""

    # Get communicator to use
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Make sure input is an array
    arr = np.asanyarray(arr)

    # Get number of elements in dimensions after the first
    nvalues = 1
    if len(arr.shape) > 1:
        for s in arr.shape[1:]:
            nvalues *= s

    # Find total number of elements
    n        = arr.shape[0]
    nperproc = comm.allgather(n)
    ntot     = sum(nperproc)

    # Check total doesn't change
    if ntot != sum(ndesired):
        print("repartition() - number of elements must be conserved!")
        comm.Abort()

    # Find first index on each processor
    first_on_proc_in  = np.cumsum(nperproc) - nperproc
    first_on_proc_out = np.cumsum(ndesired) - ndesired

    # Count elements to go to each other processor
    send_count = np.zeros(comm_size, dtype=index_dtype)
    for rank in range(comm_size):
        # Find range of elements to go to this other processor
        ifirst = first_on_proc_out[rank]
        ilast  = ifirst + ndesired[rank] - 1
        # We can only send the elements which are stored locally
        ifirst = max((ifirst, first_on_proc_in[comm_rank]))
        ilast  = min((ilast,  first_on_proc_in[comm_rank] +
                      nperproc[comm_rank] - 1))
        send_count[rank] = max((0,ilast-ifirst+1))

    # Transfer the data
    send_displ = np.cumsum(send_count) - send_count
    recv_count = np.ndarray(comm_size, dtype=index_dtype)
    comm.Alltoall(send_count, recv_count)
    recv_displ = np.cumsum(recv_count) - recv_count

    shape = list(arr.shape)
    shape[0] = sum(recv_count)
    arr_return = np.empty_like(arr, shape=shape, dtype=arr.dtype)
    my_alltoallv(arr.reshape((-1,)),        nvalues*send_count, nvalues*send_displ,
                 arr_return.reshape((-1,)), nvalues*recv_count, nvalues*recv_displ,
                 comm=comm)

    # Return the resulting new array
    return arr_return



def fetch_elements(arr, index, result=None, comm=None):
    """
    Return the specified elements from a distributed array.
    This can be used to apply the sorting index returned by
    parallel_sort().

    Input:

    arr    - the array containing the elements to retreive
    index  - index of elements to retrieve, numbered starting at zero
             for the first element on rank 0 and going up to len(arr)-1
             for the last element on rank comm_size-1.
    result - an array in which to store the result. A new array is
             returned if this is None.

    Returns:

    Array containing elements in the order specified by index,
    if result is None.
    """

    # Get communicator to use
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    arr = np.asanyarray(arr)
    index = np.asanyarray(index, dtype=index_dtype)

    # Get number of elements in dimensions after the first
    nvalues = 1
    if len(arr.shape) > 1:
        for s in arr.shape[1:]:
            nvalues *= s

    # Find total number of elements
    n           = arr.shape[0]
    nperproc_in = comm.allgather(n)
    ntot        = sum(nperproc_in)

    # Get sorting index for local requested elements
    idx = my_argsort(index)
    index = index[idx]

    # Find first index on each processor
    first_index_on_proc = np.cumsum(nperproc_in) - nperproc_in
    
    # Find first and last element to retrieve from each other processor
    ifirst = np.searchsorted(index, first_index_on_proc)
    ilast = ifirst.copy()
    ilast[:-1] = ifirst[1:] - 1
    ilast[-1] = index.shape[0] - 1

    # Send indexes to fetch to processors which have the data
    send_count = np.asarray(ilast-ifirst+1, dtype=index_dtype)
    send_displ = np.cumsum(send_count) - send_count
    recv_count = np.ndarray(comm_size, dtype=index_dtype)
    comm.Alltoall(send_count, recv_count)
    recv_displ = np.cumsum(recv_count) - recv_count
    index_find = np.ndarray(sum(recv_count), dtype=index_dtype)
    my_alltoallv(index,      send_count, send_displ,
                 index_find, recv_count, recv_displ,
                 comm=comm)

    # Find values to return
    index_find -= first_index_on_proc[comm_rank]
    values = arr[index_find,...]
    del index_find

    # Send values back
    if result is None:
        # Will return result, so allocate storage
        shape = list(arr.shape)
        shape[0] = index.shape[0]
        values_recv = np.empty_like(arr, shape=shape)
    else:
        # Will put result in supplied array
        values_recv = result

    my_alltoallv(values.reshape((-1,)),      nvalues*recv_count, nvalues*recv_displ,
                 values_recv.reshape((-1,)), nvalues*send_count, nvalues*send_displ,
                 comm=comm)
    del values

    # Restore original order using index array
    values_recv[idx,...] = values_recv.copy()

    if result is None:
        return values_recv



def weighted_median(arr, weight):
    """Return the median of array 'arr' weighted by 'weight'"""    
    # Convert input to numpy arrays if necessary
    arr = np.asanyarray(arr)
    weight = np.asanyarray(weight, dtype=np.float64)
    # Sort into ascending order
    idx    = my_argsort(arr)
    arr    = arr[idx]
    weight = weight[idx]
    # Normalize so weights add up to 1
    weight = weight / np.sum(weight)
    # Find weight of all previous entries for each element
    wbefore = np.cumsum(weight) - weight
    # Find weight of subsequent entries for each element
    wafter  = 1.0 - np.cumsum(weight)
    # Find elements with <0.5 on either side
    ind = np.logical_and(wbefore<=0.5,wafter<=0.5)
    return arr[ind][0]


def gather_to_2d_array(arr_in, comm):
    """
    Gather 1D arrays arr_in to make a new 2D array
    with dimensions [comm_size, len(arr_in)].

    arr_in needs to be the same size on all tasks
    """
    comm_size = comm.Get_size()
    mpi_type = mpi_datatype(arr_in.dtype)
    arr_out = np.empty_like(arr_in, shape=comm_size*len(arr_in), dtype=arr_in.dtype)
    comm.Allgather([arr_in, mpi_type], [arr_out, mpi_type])
    mpi_type.Free()
    return arr_out.reshape((comm_size, len(arr_in)))


def find_splitting_points(arr, r, comm=None):
    """
    Find the values in distributed array arr with global ranks r.

    In this version r can be an array so we can more efficiently
    find multiple values.
    """

    # Get communicator to use
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD

    # Get communicator size and rank
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Ensure inputs are arrays
    arr = np.asanyarray(arr)
    r = np.asanyarray(r)

    # Get number of elements per processor
    nperproc = np.asarray(comm.allgather(len(arr)))

    # Arrays with one element per rank to find
    nranks = len(r)
    imin = np.zeros(nranks, dtype=index_dtype)
    imax = np.ones( nranks, dtype=index_dtype) * (len(arr) - 1)
    done = np.zeros(nranks, dtype=bool)

    local_median = np.zeros(nranks, dtype=arr.dtype)
    local_weight = np.zeros(nranks, dtype=float)
    est_median   = np.zeros(max(nranks, comm_size), dtype=arr.dtype)

    # Arrays with result
    res_median   = np.zeros(nranks, dtype=arr.dtype)
    res_min_rank = np.zeros(nranks, dtype=index_dtype)
    res_max_rank = np.zeros(nranks, dtype=index_dtype)

    # Send/recv buffers
    send_median = np.zeros(comm_size, dtype=local_median.dtype)
    send_weight = np.zeros(comm_size, dtype=local_weight.dtype)
    recv_median = np.zeros(comm_size, dtype=local_median.dtype)
    recv_weight = np.zeros(comm_size, dtype=local_weight.dtype)

    # MPI type corresponding to the array to sort
    mpi_type = mpi_datatype(local_median.dtype)

    # Iterate until we find all splitting points
    while any(done==False):

        # Estimate median of elements in active ranges for each rank
        nlocal  = np.maximum(0,imax-imin+1)             # Array with local size of active range for each rank
        nactive_sum = np.empty_like(nlocal)
        comm.Allreduce(nlocal, nactive_sum, op=MPI.SUM)
        for i in range(nranks):
            if nlocal[i] > 0 and not(done[i]):
                local_median[i] = arr[imin[i]+nlocal[i]//2]
                local_weight[i] = float(nlocal[i]) / float(nactive_sum[i])
            else:
                local_median[i] = 0.0
                local_weight[i] = 0.0

        if nranks > comm_size:
            # This method is slow but works for any number of ranks to find
            medians = gather_to_2d_array(local_median, comm) # Returns 2D array of size [comm_size, nranks]
            weights = gather_to_2d_array(local_weight, comm)
            # Calculate median of medians for each rank
            for i in range(nranks):
                if not(done[i]):
                    est_median[i] = weighted_median(medians[:,i], weights[:,i])
                else:
                    est_median[i] = 0.0
        else:
            # if nranks <= comm_size we can assign each rank to a different MPI task
            # and compute the medians in parallel
            send_median[:nranks] = local_median
            send_weight[:nranks] = local_weight
            comm.Alltoall([send_median, mpi_type], [recv_median, mpi_type])
            comm.Alltoall(send_weight, recv_weight)
            if comm_rank < nranks and not(done[comm_rank]):
                local_est_median = weighted_median(recv_median, recv_weight)
            else:
                local_est_median = recv_median[0] # Not used, but need to set a value for Allgather
            # All MPI tasks need the full array of medians
            comm.Allgather([local_est_median, mpi_type], [est_median, mpi_type])
        
        # Find first and last elements where estimated medians can be inserted in sorted array
        ifirst = np.searchsorted(arr, est_median[:nranks][done==False], side='left')
        ilast  = np.searchsorted(arr, est_median[:nranks][done==False], side='right')
        fsum = np.empty_like(ifirst)
        comm.Allreduce(ifirst, fsum)
        lsum = np.empty_like(ilast)
        comm.Allreduce(ilast, lsum)

        j = 0 # Index into arrays which only have elements for ranks with done=False
        for i in range(nranks): # Index into arrays with elements for all ranks
            if not done[i]:
                # Check if we've found the target rank
                min_rank = fsum[j]      # Min rank est_median could have
                max_rank = lsum[j] - 1  # Max rank est_median could have
                # If requested rank is in the range of ranks with value
                # est_median, we're done.
                if r[i] >= min_rank and r[i] <= max_rank:
                    res_median[i]   = est_median[i]
                    res_min_rank[i] = min_rank
                    res_max_rank[i] = max_rank
                    done[i]         = True
                # If rank of est_median is too low, increase lower limit
                if max_rank < r[i]:
                    imin[i] = max((imin[i], ilast[j]))
                # If rank of est_median is too high, lower the upper limit
                if min_rank > r[i]:
                    imax[i] = min((imax[i], ifirst[j] - 1))
                j += 1

    mpi_type.Free()

    return res_median, res_min_rank, res_max_rank


def parallel_sort(arr, comm=None, return_index=False, verbose=False):
    """
    Sort the input array in place using an iterative scheme to split
    values between processors.
    
    Input:

    arr - array which will be sorted in place

    Returns: 

    Sorting index if return_index=True, otherwise None
    """

    # Get communicator to use
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD

    # Ensure input is an array. Note that we don't use np.asanyarray here because
    # if arr is an ndarray subclass we want to access the underlying ndarray for
    # efficiency. This function doesn't return any values of the same type as the
    # array so this does not affect the output.
    arr = np.asarray(arr)

    # mpi4py doesn't like wrong endian data!
    if not(arr.dtype.isnative):
        print("parallel_sort.py: Unable to operate on non-native endian data!")
        comm.Abort()

    # Record starting time
    if verbose:
        comm.Barrier()
        t0 = time.time()

    # Sanity check input
    if not(hasattr(arr,"dtype")) or not(hasattr(arr,"shape")):
        print("Can only sort arrays!")
        comm.Abort()
    if len(arr.shape) != 1:
        print("Can only sort 1D data!")
        comm.Abort()
        
    # Make a new communicator containing only processors
    # which have data so we don't have to worry about empty
    # arrays.
    if arr.shape[0] > 0:
        colour = 1
    else:
        colour = 0
    key    = comm.Get_rank()
    mycomm = comm.Split(colour, key)
    mycomm_rank = mycomm.Get_rank()
    mycomm_size = mycomm.Get_size()    

    # If we have no data, there's nothing to do
    if arr.shape[0] > 0:

        # Find total number of elements
        n           = arr.shape[0]
        nperproc    = mycomm.allgather(n)
        ntot        = np.sum(nperproc)

        # Sort array locally
        if verbose and mycomm_rank==0:
            print("Sorting local elements, t = ", time.time()-t0)
        sort_idx1 = my_argsort(arr)
        arr[:] = arr[sort_idx1]
        if not(return_index):
            del sort_idx1

        # Find global rank of first value to put on each processor
        split_rank = np.cumsum(nperproc) - nperproc

        # Find values at which we want to split the sorted
        # array and also minimum and maximum global ranks
        # of these values (first element on processor mycomm_rank=0 has
        # global rank 0)
        if verbose and mycomm_rank==0:
            print("Calculating splitting points, t = ", time.time()-t0)

        # Find the values with ranks in split_rank, and also the maximum and
        # minimum ranks which have this value
        val, min_rank, max_rank = find_splitting_points(arr, split_rank, mycomm)
        split_instance = split_rank - min_rank

        # Find out how many instances of each splitting value we have
        # on each processor
        if arr.shape[0] > 0:
            first_ind  = np.searchsorted(arr, val, side='left')
            last_ind   = np.searchsorted(arr, val, side='right') - 1
            nval_local = last_ind - first_ind + 1
            first_ind[first_ind >= len(arr)] = len(arr) - 1
            nval_local[arr[first_ind] != val] = 0 
        else:
            nval_local = np.zeros(len(val), dtype=index_dtype)
        nval_local = np.asarray(nval_local, dtype=index_dtype)

        # Find out how many instances of each value there are on each
        # lower ranked processor - indexes are (iproc, ival) after gather.
        nval_all = np.ndarray((mycomm_size, mycomm_size), dtype=index_dtype)
        mycomm.Allgather(nval_local, nval_all) 
        nval_lower = np.zeros(len(val), dtype=index_dtype)
        for ival in range(len(val)):
            if mycomm_rank > 0:
                nval_lower[ival] = sum(nval_all[0:mycomm_rank,ival])
            else:
                nval_lower[ival] = 0

        # Determine which local elements are to go to which other
        # processor
        if verbose and mycomm_rank==0:
            print("Determining destination for each element, t = ", time.time()-t0)
        first_to_send = 0
        send_count = np.zeros(mycomm_size, dtype=index_dtype)

        # Find array indexes of splitting point values
        # It's much faster to do this in one pass here than to do
        # each one separately inside the loop below.
        val_first_index_arr = np.searchsorted(arr, val, side='left')
        val_last_index_arr  = np.searchsorted(arr, val, side='right') - 1

        # Loop over destination processors
        for rank in range(mycomm_size):
            # May be no elements to go to this processor
            if nperproc[rank] == 0 or first_to_send >= arr.shape[0]:
                continue
            # At least one element, starting at first_to_send,
            # needs to go to processor specified by rank.
            # Need to find how many to send.
            if rank < mycomm_size - 1:
                # Find index of first value to go to the next
                # processor (note that some *instances* of this
                # value may go to the current processor)
                val_first_index = val_first_index_arr[rank+1]
                val_last_index  = val_last_index_arr[rank+1]
                if val_first_index > arr.shape[0] - 1:
                    # First value for next processor doesn't exist
                    # locally, so all remaining elements go to this
                    # processor
                    last_to_send = arr.shape[0] - 1
                else:
                    if arr[val_first_index_arr[rank+1]] != val[rank+1]:
                        # All remaining instances of this value go to later
                        # processors
                        last_to_send = val_first_index - 1
                    else:
                        # In this case element val_first_index is the first
                        # local instance of the first value to go to the next
                        # processor.
                        # Find what range of instances of this value we have locally
                        val_first_instance = nval_lower[rank+1]
                        val_last_instance  = nval_lower[rank+1] + (val_last_index-val_first_index)
                        # Decide how many instances go to this processor
                        if split_instance[rank+1] > val_last_instance:
                            # Splitting point is after last instance we have locally,
                            # so send them all
                            last_to_send = val_last_index
                        elif split_instance[rank+1] <= val_first_instance:
                            # Splitting point is before the first instance we have locally,
                            # so send none to this processor
                            last_to_send = val_first_index - 1
                        else:
                            # Splitting point lies in the range we have stored locally.
                            # Send only instances < split_instance[rank+1]
                            nsend = split_instance[rank+1] - val_first_instance
                            last_to_send = val_first_index + nsend - 1
            else:
                # Send all remaining elements to last processor
                last_to_send = arr.shape[0] - 1
            # Record number of elements to send 
            send_count[rank] = max((0,last_to_send - first_to_send + 1))
            # Advance to elements for next processor
            first_to_send = max((first_to_send, last_to_send + 1))

        # Calculate counts and offsets for alltoallv
        send_displ = np.cumsum(send_count) - send_count
        recv_count = np.ndarray(mycomm_size, dtype=index_dtype)
        mycomm.Alltoall(send_count, recv_count)
        recv_displ = np.cumsum(recv_count) - recv_count

        # Exchange data
        if verbose and mycomm_rank==0:
            print("Exchanging data, t = ", time.time()-t0)
        arr_tmp = np.empty_like(arr, shape=sum(recv_count))
        my_alltoallv(arr,     send_count, send_displ,
                     arr_tmp, recv_count, recv_displ,
                     comm=mycomm)

        # Sort local data
        if verbose and mycomm_rank==0:
            print("Sorting local elements, t = ", time.time()-t0)
        sort_idx2 = my_argsort(arr_tmp)
        arr[:] = arr_tmp[sort_idx2]
        del arr_tmp

        #
        # Generate index array if necessary.
        # This is done by moving the index values around in the
        # same way we did the data array, which requires that we keep
        # the sorting indexes from the initial and final local sorts
        # and the start/count arrays from the alltoallv call.
        #
        if return_index:
            if verbose and mycomm_rank==0:
                print("Making index array, t = ", time.time()-t0)
            # Rearrange indices using index from initial local sort
            # of the data array
            index = np.arange(arr.shape[0], dtype=index_dtype)
            if mycomm_rank > 0:
                index[:] += np.sum(nperproc[0:mycomm_rank], dtype=index_dtype)
            index = index[sort_idx1]
            del sort_idx1
            # Exchange index data in same way as array values
            index_tmp = np.ndarray(sum(recv_count), dtype=index_dtype)
            my_alltoallv(index,     send_count, send_displ,
                         index_tmp, recv_count, recv_displ,
                         comm=mycomm)

            # Reorder local index data using index from final local sort
            index[:] = index_tmp[sort_idx2]
            del index_tmp

        # Done with local sorting index
        del sort_idx2

    else:
        if return_index:
            # No elements here, so return empty array
            index = np.ndarray(0, dtype=index_dtype)

    if verbose and mycomm_rank==0:
        print("Parallel sort finished, t = ", time.time()-t0)

    # Finished with communicator
    mycomm.Free()


    # Return index array if necessary
    if return_index:
        return index
    else:
        return None


def parallel_match(arr1, arr2, arr2_sorted=False, comm=None):
    """
    For each element in arr1 return the global index of an
    element with the same value in arr2, or -1 if there's
    no element with the same value.

    If arr2_sorted=True we assume that a *parallel* sort has
    already been done on arr2. This saves time if this function is
    called repeatedly with the same arr2.
    """

    # Get communicator to use
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    # Ensure inputs are array-like
    arr1 = np.asanyarray(arr1)
    arr2 = np.asanyarray(arr2)

    # Sanity checks on input arrays
    if len(arr1.shape) != 1 or len(arr2.shape) != 1:
        print("Can only match elements between 1D arrays!")
        comm.Abort()

    # If arr1 has no elements we just return an empty index array
    if comm.allreduce(arr1.shape[0]) == 0:
        return np.ndarray(0, dtype=index_dtype)

    # If arr2 has no elements then nothing will match
    if comm.allreduce(arr2.shape[0]) == 0:
        return -np.ones(arr1.shape, dtype=index_dtype)

    # Make a sorted copy of arr2 if necessary
    if not(arr2_sorted):
        arr2_ordered = arr2.copy()
        sort_arr2 = parallel_sort(arr2_ordered, return_index=True, comm=comm)
    else:
        arr2_ordered = arr2

    # Make array of initial indexes of arr2 elements and arrange in sorted order
    n_per_task = np.asarray(comm.allgather(arr2.shape[0]), dtype=index_dtype)
    first_on_task = np.cumsum(n_per_task) - n_per_task
    index_in = np.arange(arr2.shape[0], dtype=index_dtype) + first_on_task[comm_rank]
    if not(arr2_sorted):
        index_in = fetch_elements(index_in, sort_arr2, comm=comm)
        del sort_arr2

    # Find range of arr2 values on each task after sorting
    min_on_task = np.ndarray(comm_size, dtype=arr2.dtype)
    max_on_task = np.ndarray(comm_size, dtype=arr2.dtype)
    if arr2_ordered.shape[0] > 0:
        min_val = np.amin(arr2_ordered)
        max_val = np.amax(arr2_ordered)
    else:
        # If there are no values set min and max to zero
        min_val = np.asarray(0, dtype=arr2_ordered.dtype)
        max_val = np.asarray(0, dtype=arr2_ordered.dtype)
    comm.Allgather(min_val, min_on_task)
    comm.Allgather(max_val, max_on_task)
    # Record which tasks have >0 arr2 elements
    have_arr2 = np.asarray(comm.allgather(arr2_ordered.shape[0] > 0), dtype=bool)

    # Sort local arr1 values
    idx = my_argsort(arr1)
    arr1_ls = arr1[idx]
    
    # Decide which elements of arr1 to send to which tasks
    # Handle duplicate values in arr2 by sending to lowest numbered task.
    # First we calculate offsets and counts for tasks with >0 elements in arr2.
    send_displ      = np.searchsorted(arr1_ls, min_on_task[have_arr2], side="left")
    send_displ_next = np.searchsorted(arr1_ls, max_on_task[have_arr2], side="right")
    if sum(have_arr2) > 1:
        send_displ[1:] = send_displ_next[:-1]
    send_count = send_displ_next - send_displ
    send_count[send_count<0] = 0
    assert np.sum(send_count) <= arr1_ls.shape[0]
    assert all(send_displ >= 0)
    assert all(send_displ+send_count <= arr1_ls.shape[0])

    # Expand displacement and count arrays to include tasks with no arr2 elements
    send_displ_all = np.zeros(comm_size, dtype=send_displ.dtype)
    send_displ_all[have_arr2] = send_displ
    send_count_all = np.zeros(comm_size, dtype=send_count.dtype)
    send_count_all[have_arr2] = send_count # Needs to be zero where have_arr2 is false
    send_count = send_count_all
    send_displ = send_displ_all

    # Transfer the data
    recv_count = np.ndarray(comm_size, dtype=index_dtype)
    comm.Alltoall(send_count, recv_count)
    recv_displ = np.cumsum(recv_count) - recv_count
    arr1_recv = np.ndarray(np.sum(recv_count), dtype=arr1.dtype)
    my_alltoallv(arr1_ls,   send_count, send_displ,
                 arr1_recv, recv_count, recv_displ,
                 comm=comm)
    del arr1_ls

    # For each imported arr1 element, find global rank of matching arr2 element
    ptr = np.searchsorted(arr2_ordered, arr1_recv, side="left")
    ptr[ptr<0] = 0
    ptr[ptr>=arr2_ordered.shape[0]] = 0 
    ptr[arr2_ordered[ptr] != arr1_recv] = -1
    del arr1_recv
    del arr2_ordered
    index_return = -np.ones(ptr.shape, dtype=index_dtype)
    index_return[ptr>=0] = index_in[ptr[ptr>=0]]
    del index_in
    del ptr

    # Return the index info
    index_out = np.zeros(arr1.shape, dtype=index_dtype) - 1
    my_alltoallv(index_return, recv_count, recv_displ,
                 index_out,    send_count, send_displ,
                 comm=comm)
    del index_return

    # Restore original order
    index = np.empty_like(index_out)
    index[idx] = index_out
    del index_out
    del idx

    return index


def parallel_unique(arr, comm=None, arr_sorted=False, return_counts=False,
                    repartition_output=False):
    """
    Given a distributed array arr, return a new distributed
    array which contains the sorted, unique elements in arr.

    If return counts is true then also return another
    distributed array with the number of instances of each
    unique value.

    If arr has already been parallel sorted, can set arr_sorted=True
    to save some time. This will produce incorrect results if the
    array has not been parallel sorted.

    If repartition is True, try to leave similar numbers of
    output elements on each MPI rank.
    """

    # Get communicator to use
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Ensure input is an array
    arr = np.asanyarray(arr)

    # Sanity checks on input arrays
    if len(arr.shape) != 1:
        print("Can only find unique elements in 1D arrays!")
        comm.Abort()        

    # Make a new communicator containing only processors
    # which have data so we don't have to worry about empty
    # arrays.
    if arr.shape[0] > 0:
        colour = 1
    else:
        colour = 0
    key    = comm.Get_rank()
    mycomm = comm.Split(colour, key)
    mycomm_rank = mycomm.Get_rank()
    mycomm_size = mycomm.Get_size()    

    if colour == 1:

        # Make a sorted copy of the input if necessary
        if not arr_sorted:
            arr = arr.copy()
            parallel_sort(arr, comm=mycomm)

        # Find unique elements on each rank
        local_unique, local_count = np.unique(arr, return_counts=True)
        del arr

        # Gather value and count for first and last values on each rank
        all_first_unique = mycomm.allgather(local_unique[0])
        all_first_counts = mycomm.allgather(local_count[0])
        all_last_unique = mycomm.allgather(local_unique[-1])
        all_last_counts = mycomm.allgather(local_count[-1])

        # Accumulate counts from later ranks which have instances of our last unique value
        rank_nr = mycomm_rank + 1
        while rank_nr < mycomm_size and all_first_unique[rank_nr] == all_last_unique[mycomm_rank]:
            local_count[-1] += all_first_counts[rank_nr]
            rank_nr += 1

        # If our first unique value exists on the previous rank, zero out our count for this value
        if mycomm_rank > 0 and all_first_unique[mycomm_rank] == all_last_unique[mycomm_rank-1]:
            local_count[0] = 0
            
        # Discard values with zero count
        keep = local_count > 0
        local_unique = local_unique[keep]
        local_count = local_count[keep]

    else:
        # This rank has no elements
        local_unique = np.zeros(0, dtype=arr.dtype)
        local_count = np.zeros(0, dtype=int)

    mycomm.Free()

    # Repartition if necessary
    total_nr_unique = comm.allreduce(len(local_unique))
    if repartition_output:
        ndesired = np.zeros(comm_size, dtype=int)
        ndesired[:] = total_nr_unique // comm_size
        ndesired[:total_nr_unique % comm_size] += 1
        assert ndesired.sum() == total_nr_unique
        local_unique = repartition(local_unique, ndesired, comm=comm)
        local_count = repartition(local_count, ndesired, comm=comm)

    # Return the results
    if return_counts:
        return local_unique, local_count
    else:
        return local_unique


def gather_to_first_rank(arr, comm):
    """Gather the specified array on rank 0"""
    arr_g = comm.gather(arr)
    if comm.Get_rank() == 0:
        return np.concatenate(arr_g)
    else:
        return None


def reduce_elements(arr, updates, index, op, comm=None):
    """
    Update the elements given by index of distributed array arr by
    reducing them with the values in updates using MPI operator op.

    For example, on one MPI rank with op=MPI.SUM this is equivalent
    to

    arr[index,...] += updates

    On multiple MPI ranks the index is global and may refer to
    elements of arr which are stored on other ranks.

    For multidimensional arrays index is taken to refer to the first
    dimension.

    The operator may alternatively be a numpy ufunc. In this case
    its at() method is used to apply the updates, which is likely to
    be faster.
    """

    # Get communicator to use
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD

    # Ensure input indexes are an array of index_dtype
    index = np.asarray(index, dtype=index_dtype)

    # Check that index is 1D
    if len(index.shape) != 1:
        print("update_elements() - index must be one dimensional!")
        comm.Abort()

    # Check that we have the expected number of updates
    if index.shape[0] != updates.shape[0]:
        print("update_elements() - index and updates must have same first dimension!")
        comm.Abort()

    # Check that updates have the same shape as arr (except first dimension)
    if updates.shape[1:] != arr.shape[1:]:
        print("update_elements() - arr and updates have inconsistent shapes!")
        comm.Abort()

    # Find the range of global indexes of arr on each rank
    local_offset = comm.scan(len(arr)) - len(arr)
    local_length = len(arr)
    all_local_offsets = np.asarray(comm.allgather(local_offset), dtype=index_dtype)
    all_local_lengths = np.asarray(comm.allgather(local_length), dtype=index_dtype)

    # Sort updates by destination index
    order   = np.argsort(index)
    index   = index[order]
    updates = updates[order]
    del order

    # Find which rank each update needs to go to
    send_offset = np.searchsorted(index, all_local_offsets, side="left")
    send_count = np.zeros_like(send_offset)
    send_count[:-1] = send_offset[1:] - send_offset[:-1]
    send_count[-1] = len(index) - send_offset[-1]
    assert sum(send_count) == len(index)

    # Compute send and receive counts
    send_displ = np.cumsum(send_count) - send_count
    recv_count = np.ndarray(comm.Get_size(), dtype=index_dtype)
    comm.Alltoall(send_count, recv_count)
    recv_displ = np.cumsum(recv_count) - recv_count

    # Exchange indexes to update
    index_recv = np.empty_like(index, shape=sum(recv_count))
    my_alltoallv(index,      send_count, send_displ,
                 index_recv, recv_count, recv_displ,
                 comm=comm)

    # Compute number of values per index (in case of multidimensional input)
    nvalues = 1
    for s in updates.shape[1:]:
        nvalues *= s

    # Exchange updates
    recv_shape = (sum(recv_count),) + updates.shape[1:]
    updates_recv = np.empty_like(updates, shape=recv_shape)
    my_alltoallv(updates.reshape((-1,)),      nvalues*send_count, nvalues*send_displ,
                 updates_recv.reshape((-1,)), nvalues*recv_count, nvalues*recv_displ,
                 comm=comm)

    # Convert received array indexes to local indexes
    index_recv -= local_offset
    assert np.all(index_recv >= 0)
    assert np.all(index_recv <= len(arr))

    # Apply updates to the local array elements
    if isinstance(op, np.ufunc):
        # op is a numpy ufunc
        op.at(arr, index_recv, updates_recv)
    else:
        # op is an MPI operator
        for i, j in enumerate(index_recv):
            op.Reduce_local(updates_recv[i,...], arr[j,...])


def parallel_bincount(x, weights=None, minlength=None, result=None, comm=None):
    """
    Parallel version of numpy.bincount where input and output are
    distributed arrays.
    """

    # Get communicator to use
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    # Ensure inputs are arrays
    x = np.asanyarray(x)
    if weights is not None:
        weights = np.asanyarray(weights)
    
    # Input needs to be integer
    if x.dtype.kind != "i" and x.dtype.kind != "u":
        raise ValueError("Input array x must be an integer type")

    # Determine output array data type
    if result is not None:
        result_dtype = result.dtype
    elif weights is not None:
        result_dtype = weights.dtype
    else:
        result_dtype = index_dtype

    # Find the maximum value in x
    x_max = np.amax(x) if len(x) > 0 else 0
    x_max = comm.allreduce(x_max, op=MPI.MAX)
    total_nr_bins = x_max+1

    # Find total number of bins needed
    if minlength is not None:
        total_nr_bins = int(max(total_nr_bins, minlength))

    # Allocate result, if necessary
    if result is None:
        # Need to decide how to distribute result array if it wasn't provided
        local_nr_bins = total_nr_bins // comm_size
        if comm_rank < total_nr_bins % comm_size:
            local_nr_bins += 1
        result = np.zeros(local_nr_bins, dtype=result_dtype)
    else:
        # Just check result array is large enough
        if comm.allreduce(len(result)) < total_nr_bins:
            raise ValueError("Result array is too small")
    
    # If no weights are specified, set all to 1
    if weights is None:
        weights = np.ones(len(x), dtype=index_dtype)

    # Sort values and weights
    order = np.argsort(x)
    x = x[order]
    weights = weights[order]
        
    # Now combine local elements
    unique_x, unique_indices, unique_counts = np.unique(x, return_index=True, return_counts=True)
    unique_weights = np.empty_like(weights, shape=unique_x.shape)
    for i, (offset, length) in enumerate(zip(unique_indices, unique_counts)):
        unique_weights[i] = np.sum(weights[offset:offset+length])
    
    # Accumulate weights over all MPI ranks
    reduce_elements(result, unique_weights, unique_x, op=np.add, comm=comm)

    return result


class HashMatcher:
    """
    Class for matching integers between pairs of arrays.

    This is similar to parallel_match(arr1, arr2) in that for each element
    in arr1 it returns the index of a matching element in arr2, or -1 if no
    match is found. Currently only implemented for 4 or 8 byte types.

    Instantiating the class redistributes arr2 over MPI ranks so that we
    can carry out repeated searches of arr2 with different arr1.

    Best not to use python's hash() here because hash(n)=n for integers.
    """

    def destination_rank(self, arr):
        if arr.dtype.itemsize == 4:
            arr_view = arr.view(dtype=np.uint32)
            arr_hash = np.bitwise_xor(np.right_shift(arr_view, 16), arr_view) * 0x45d9f3b
            arr_hash = np.bitwise_xor(np.right_shift(arr_hash, 16), arr_hash) * 0x45d9f3b
            arr_hash = np.bitwise_xor(np.right_shift(arr_hash, 16), arr_hash)
        elif arr.dtype.itemsize == 8:
            arr_view = arr.view(dtype=np.uint64)
            arr_hash = np.bitwise_xor(arr_view, np.right_shift(arr_view, 30)) * 0xbf58476d1ce4e5b9
            arr_hash = np.bitwise_xor(arr_hash, np.right_shift(arr_hash, 27)) * 0x94d049bb133111eb
            arr_hash = np.bitwise_xor(arr_hash, np.right_shift(arr_hash, 31))
            return np.mod(arr_hash, self.comm_size).astype(int)
        else:
            raise RuntimeError("Unsupported data type: must be 4 or 8 bytes per element")
        
    def __init__(self, arr2, comm=None):
        """
        Initialize a new hash matcher.

        arr2 - the array of values we will be matching to
        comm - the MPI communicator to use

        This moves array elements to an MPI rank based on the hash of their
        value so that we can efficiently search for values later.
        """

        # Get communicator to use
        from mpi4py import MPI
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm
        self.comm_rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()
        
        # Get destination rank for each element
        arr2 = np.asanyarray(arr2)
        arr2_dest = self.destination_rank(arr2)

        # Find global index of first arr2 element on each rank:
        # This is just the number of elements on previous ranks.
        arr2_global_offset = self.comm.scan(len(arr2)) - len(arr2)

        # Get sorting order by destination
        arr2_order = my_argsort(arr2_dest)
    
        # Find range of elements to go to each destination when sorted by destination
        arr2_send_count  = np.bincount(arr2_dest, minlength=self.comm_size)
        arr2_send_offset = np.cumsum(arr2_send_count) - arr2_send_count
        del arr2_dest
    
        # Get arr2 global indexes ordered by destination
        arr2_index = np.arange(len(arr2), dtype=index_dtype) + arr2_global_offset
        arr2_index = arr2_index[arr2_order]

        # Get arr2 values ordered by destination
        arr2 = arr2[arr2_order]
        del arr2_order

        # Move arr2 values and indexes to destination
        self.arr2 = alltoall_exchange(arr2, arr2_send_count, comm=self.comm)
        self.arr2_index = alltoall_exchange(arr2_index, arr2_send_count, comm=self.comm)
        assert np.all(self.destination_rank(self.arr2)==self.comm_rank)

        # Sort arr2 values by value
        order = np.argsort(self.arr2)
        self.arr2 = self.arr2[order]
        self.arr2_index = self.arr2_index[order]
        
    def match(self, arr1):
        """
        Return the global index in arr2 of each value in arr1
        """
        
        # Ensure inputs are array-like
        arr1 = np.asanyarray(arr1)

        # Get destination rank for each element
        arr1_dest = self.destination_rank(arr1)

        # Get sorting order by destination
        arr1_order = np.argsort(arr1_dest)

        # Find range of elements to go to each destination when sorted by destination
        arr1_send_count  = np.bincount(arr1_dest, minlength=self.comm_size)
        arr1_send_offset = np.cumsum(arr1_send_count) - arr1_send_count
        del arr1_dest

        # Allocate output array
        match_index = -np.ones(len(arr1), dtype=index_dtype)

        # Construct sorted send buffer for arr1 elements
        arr1_sendbuf = arr1[arr1_order]

        # Exchange arr1 elements
        arr1_recvbuf = alltoall_exchange(arr1_sendbuf, arr1_send_count, comm=self.comm)
        assert np.all(self.destination_rank(arr1_recvbuf)==self.comm_rank)

        # For each imported arr1 element, find the index of the matching arr2 element
        import virgo.util.match
        ptr = virgo.util.match.match(arr1_recvbuf, self.arr2, arr2_sorted=True)
        arr1_recvbuf_index = -np.ones(arr1_recvbuf.shape, dtype=index_dtype)
        arr1_recvbuf_index[ptr>=0] = self.arr2_index[ptr[ptr>=0]]

        # Reverse exchange the indexes
        arr1_sendbuf_index =  alltoall_exchange(arr1_recvbuf_index, arr1_send_count, comm=self.comm, reverse=True)

        # Return the result in the same order as arr1
        match_index = -np.ones(len(arr1), dtype=index_dtype)
        match_index[arr1_order] = arr1_sendbuf_index

        return match_index
