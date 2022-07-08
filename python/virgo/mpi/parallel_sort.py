#!/bin/env python

from __future__ import print_function

import numpy as np

import time
import sys
import gc

# Type to use for global indexes
index_dtype = np.int64


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
                 comm=None):
    """
    Alltoallv implemented using sendrecv calls. Avoids problems
    caused when some ranks send or receive more than 2^31
    elements by splitting communications.
    """
    
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

    # Loop over pairs of processes and send data
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
    weight       = np.zeros(nranks, dtype=float)
    est_median   = np.zeros(nranks, dtype=arr.dtype)

    # Arrays with result
    res_median   = np.zeros(nranks, dtype=arr.dtype)
    res_min_rank = np.zeros(nranks, dtype=index_dtype)
    res_max_rank = np.zeros(nranks, dtype=index_dtype)

    # Iterate until we find all splitting points
    while any(done==False):

        # Estimate median of elements in active ranges for each rank
        nlocal  = np.maximum(0,imax-imin+1)             # Array with local size of active range for each rank
        nactive     = gather_to_2d_array(nlocal, comm)
        nactive_sum = np.sum(nactive, axis=0)              # sum over tasks, gives one element per rank
        for i in range(nranks):
            if nlocal[i] > 0 and not(done[i]):
                local_median[i] = arr[imin[i]+nlocal[i]//2]
                weight[i]       = float(nlocal[i]) / float(nactive_sum[i])
            else:
                local_median[i] = 0.0
                weight[i]       = 0.0

        # Gather medians from all tasks
        medians = gather_to_2d_array(local_median, comm) # Returns 2D array of size [comm_size, nranks]
        weights = gather_to_2d_array(weight, comm)

        # Calculate median of medians for each rank
        for i in range(nranks):
            if not(done[i]):
                est_median[i] = weighted_median(medians[:,i], weights[:,i])
            else:
                est_median[i] = 0.0

        # Find first and last elements where estimated medians
        # can be inserted in sorted array
        ifirst = np.searchsorted(arr, est_median, side='left')
        ilast  = np.searchsorted(arr, est_median, side='right')
        ifirst_all = gather_to_2d_array(ifirst, comm) # Returns 2D array of size [comm_size, nranks]
        ilast_all  = gather_to_2d_array(ilast, comm)
        fsum = np.sum(ifirst_all, axis=0, dtype=index_dtype) # Sum over tasks, leaves one element per rank to find
        lsum = np.sum(ilast_all,  axis=0, dtype=index_dtype)

        for i in range(nranks):
            if not done[i]:
                # Check if we've found the target rank
                min_rank = fsum[i]      # Min rank est_median could have
                max_rank = lsum[i] - 1  # Max rank est_median could have
                # If requested rank is in the range of ranks with value
                # est_median, we're done.
                if r[i] >= min_rank and r[i] <= max_rank:
                    res_median[i]   = est_median[i]
                    res_min_rank[i] = min_rank
                    res_max_rank[i] = max_rank
                    done[i]         = True
                # If rank of est_median is too low, increase lower limit
                if max_rank < r[i]:
                    imin[i] = max((imin[i], ilast[i]))
                # If rank of est_median is too high, lower the upper limit
                if min_rank > r[i]:
                    imax[i] = min((imax[i], ifirst[i] - 1))

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


def test_parallel_sort(input_function, nr_tests, message):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if comm_rank == 0:
        print(f"Test sorting {nr_tests} {message}")

    for i in range(nr_tests):

        # Make the test array
        arr = input_function()

        # Parallel sort then gather result on rank 0
        arr_ps = arr.copy()
        index = parallel_sort(arr_ps, return_index=True)
        arr_ps_g = gather_to_first_rank(arr_ps, comm)

        # Gather on rank 0 then serial sort
        arr_g = gather_to_first_rank(arr, comm)
        if comm_rank == 0:
            arr_g_ss = np.sort(arr_g)

        # Compare
        if comm_rank == 0:
            if np.any(arr_ps_g != arr_g_ss):
                raise RuntimeError("FAILED: array was not sorted correctly!")

        # Check that we can reconstruct the sorted array using the index
        arr_ps_from_index = fetch_elements(arr, index)
        if np.any(arr_ps_from_index != arr_ps):
            raise RuntimeError("FAILED: Index doesn't work!")

    comm.Barrier()
    if comm_rank == 0:
        print(f"  OK")


def test_parallel_sort_random_integers():

    def input_function():
        max_local_size = 10000
        max_value = 10
        n   = np.random.randint(max_local_size) + 0
        arr = np.random.randint(max_value, size=n)
        return arr

    test_parallel_sort(input_function, 200, "random arrays of integers")


def test_parallel_sort_random_floats():

    def input_function():
        max_local_size = 10000
        max_value = 1.0e10
        n   = np.random.randint(max_local_size) + 0
        arr = np.random.uniform(low=-max_value, high=max_value, size=n)
        return arr

    test_parallel_sort(input_function, 200, "random arrays of floats")


def test_parallel_sort_all_empty():

    def input_function():
        return np.zeros(0, dtype=float)
    test_parallel_sort(input_function, 1, "empty arrays")


def test_parallel_sort_some_empty():

    def input_function():
        max_local_size = 10000
        max_value = 1.0e10
        n   = np.random.randint(max_local_size) + 0
        if np.random.randint(2) == 0:
            n = 0 # 50% chance for rank to have empty array
        arr = np.random.uniform(low=-max_value, high=max_value, size=n)
        return arr

    test_parallel_sort(input_function, 200, "arrays with some empty ranks")


def test_parallel_sort_unyt_floats():

    try:
        import unyt
    except ImportError:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Skipping unyt test: failed to import unyt")
        return

    def input_function():
        max_local_size = 10000
        max_value = 1.0e10
        n   = np.random.randint(max_local_size) + 0
        arr = np.random.uniform(low=-max_value, high=max_value, size=n)
        return unyt.unyt_array(arr, units=unyt.cm)

    test_parallel_sort(input_function, 200, "random unyt float arrays")


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

    test_parallel_sort(input_function, 200, "structured arrays")

  
def test_repartition_random_integers():

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    nr_tests = 200
    max_local_size = 1000
    max_value = 20

    if comm_rank == 0:
        print(f"Test repartitioning {nr_tests} random integer arrays")

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
        new_arr = repartition(arr, ndesired)

        # Verify result by gathering on rank 0
        arr_gathered = np.concatenate(comm.allgather(arr))
        new_arr_gathered = np.concatenate(comm.allgather(new_arr))
        if comm_rank == 0:
            if np.any(arr_gathered != new_arr_gathered):
                raise RuntimeError("FAILED: Partitioned array is incorrect!")

    comm.Barrier()
    if comm_rank == 0:
        print(f"  OK")


def test_repartition_structured_array():

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    nr_tests = 200
    max_local_size = 1000
    max_value = 20

    dtype = np.dtype([("a", int), ("b", int)])

    if comm_rank == 0:
        print(f"Test repartitioning {nr_tests} random structured arrays")

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
        new_arr = repartition(arr, ndesired)

        # Verify result by gathering on rank 0
        arr_gathered = np.concatenate(comm.allgather(arr))
        new_arr_gathered = np.concatenate(comm.allgather(new_arr))
        if comm_rank == 0:
            if np.any(arr_gathered != new_arr_gathered):
                raise RuntimeError("FAILED: Partitioned array is incorrect!")

    comm.Barrier()
    if comm_rank == 0:
        print(f"  OK")


def test_unique():

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    nr_tests = 200
    max_size = 1000
    max_value = 50

    if comm_rank == 0:
        print(f"Test finding unique values in {nr_tests} random integer arrays")

    for test_nr in range(nr_tests):

        # Create test dataset
        nr_local_elements = np.random.randint(max_size)
        nr_total_elements = comm.allreduce(nr_local_elements)
        local_data = np.random.randint(max_value, size=nr_local_elements)

        # Find unique values
        local_unique, local_counts = parallel_unique(local_data, comm=comm, return_counts=True,
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
            if np.any(global_unique != check_unique):
                print(global_unique, check_unique)
                raise RuntimeError("FAILED: Unique values do not match!")
            if np.any(global_counts != check_counts):
                print(global_counts, check_counts)
                raise RuntimeError("FAILED: Count values do not match!")

    comm.Barrier()
    if comm_rank == 0:
        print("  OK")


def test():

    np.random.seed(comm_rank)
    try:
        test_parallel_sort_random_integers()
        test_parallel_sort_random_floats()
        test_parallel_sort_all_empty()
        test_parallel_sort_some_empty()
        test_parallel_sort_unyt_floats()
        test_parallel_sort_structured_arrays()
        test_repartition_random_integers()
        test_repartition_structured_array()
        test_unique()
    except RuntimeError as e:
        print(e)
        print("A test failed!")
        comm.Abort()


if __name__ == "__main__":

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if comm_rank == 0:
        print(f"Running tests on {comm_size} MPI ranks")

    test()

    comm.barrier()
    if comm_rank == 0:
        print(f"All tests done")
    
