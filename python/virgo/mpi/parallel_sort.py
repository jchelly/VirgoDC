#!/bin/env python

from numpy  import *
from mpi4py import MPI

import time
import sys

# Type to use for global indexes
index_dtype = int64
    
def my_alltoallv(sendbuf, send_count, send_offset,
                 recvbuf, recv_count, recv_offset,
                 comm=None):
    """Alltooallv implemented using sendrecv calls"""
    
    # Get communicator to use
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    ptask = 0
    while(2**ptask < comm_size):
        ptask += 1

    # Loop over pairs of processes and send data
    for ngrp in range(2**ptask):
        rank = comm_rank ^ ngrp
        if rank < comm_size:
            soff = send_offset[rank]
            sc   = send_count[rank]
            roff = recv_offset[rank]
            rc   = recv_count[rank]
            if sc > 0 or rc > 0:
                comm.Sendrecv(sendbuf[soff:soff+sc], rank, 0,
                              recvbuf[roff:roff+rc], rank, 0)


        
def my_argsort(arr):
    """
    Determines serial sorting algorithm used.
    """
    # Use mergesort because
    # - it's stable (affects behaviour of parallel_match if duplicate values exist)
    # - it's worst case performance is reasonable
    return argsort(arr, kind='mergesort')



def repartition(arr, ndesired, comm=None):
    """Return the input arr repartitioned between processors"""

    # Get communicator to use
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # mpi4py doesn't like wrong endian data!
    if not(arr.dtype.isnative):
        print "parallel_sort.py: Unable to operate on non-native endian data!"
        comm.Abort()

    # Make sure input is an array
    arr   = asarray(arr)

    # Sanity checks on input arrays
    #if len(arr.shape) != 1:
    #    print "Can only repartition 1D array data!"
    #    comm.Abort()

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
        print "repartition() - number of elements must be conserved!"
        comm.Abort()

    # Find first index on each processor
    first_on_proc_in  = cumsum(nperproc) - nperproc
    first_on_proc_out = cumsum(ndesired) - ndesired

    # Count elements to go to each other processor
    send_count = zeros(comm_size, dtype=index_dtype)
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
    send_displ = cumsum(send_count) - send_count
    recv_count = ndarray(comm_size, dtype=index_dtype)
    comm.Alltoall(send_count, recv_count)
    recv_displ = cumsum(recv_count) - recv_count

    shape = list(arr.shape)
    shape[0] = sum(recv_count)
    arr_return = ndarray(shape, dtype=arr.dtype)
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
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # mpi4py doesn't like wrong endian data!
    if not(arr.dtype.isnative):
        print "parallel_sort.py: Unable to operate on non-native endian data!"
        comm.Abort()

    arr   = asarray(arr)
    index = asarray(index, dtype=index_dtype)

    # Sanity checks on input arrays
    #if len(arr.shape) != 1 or len(index.shape) != 1:
    #    print "Can only fetch elements from 1D array data!"
    #    comm.Abort()

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
    first_index_on_proc = cumsum(nperproc_in) - nperproc_in
    
    # Find first and last element to retrieve from each other processor
    ifirst = searchsorted(index, first_index_on_proc)
    ilast = ifirst.copy()
    ilast[:-1] = ifirst[1:] - 1
    ilast[-1] = index.shape[0] - 1

    # Send indexes to fetch to processors which have the data
    send_count = asarray(ilast-ifirst+1, dtype=index_dtype)
    send_displ = cumsum(send_count) - send_count
    recv_count = ndarray(comm_size, dtype=index_dtype)
    comm.Alltoall(send_count, recv_count)
    recv_displ = cumsum(recv_count) - recv_count
    index_find = ndarray(sum(recv_count), dtype=index_dtype)
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
        values_recv = ndarray(shape, dtype=arr.dtype)
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
    arr = asarray(arr)
    weight = asarray(weight, dtype=float64)
    # Sort into ascending order
    idx    = my_argsort(arr)
    arr    = arr[idx]
    weight = weight[idx]
    # Normalize so weights add up to 1
    weight = weight / sum(weight)
    # Find weight of all previous entries for each element
    wbefore = cumsum(weight) - weight
    # Find weight of subsequent entries for each element
    wafter  = 1.0 - cumsum(weight)
    # Find elements with <0.5 on either side
    ind = logical_and(wbefore<=0.5,wafter<=0.5)
    return arr[ind][0]


def gather_to_2d_array(arr_in, comm):
    """
    Gather 1D arrays arr_in to make a new 2D array
    with dimensions [comm_size, len(arr_in)].

    arr_in needs to be the same size on all tasks
    """
    comm_size = comm.Get_size()
    arr_out = ndarray(comm_size*len(arr_in), dtype=arr_in.dtype)
    comm.Allgather(arr_in, arr_out)
    return arr_out.reshape((comm_size, len(arr_in)))


def find_splitting_points(arr, r, comm=None):
    """
    Find the values in distributed array arr with global ranks r.

    In this version r can be an array so we can more efficiently
    find multiple values.
    """

    # Get communicator to use
    if comm is None:
        comm = MPI.COMM_WORLD

    # Get communicator size and rank
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Get number of elements per processor
    nperproc = asarray(comm.allgather(len(arr)))

    # Arrays with one element per rank to find
    nranks = len(r)
    imin = zeros(nranks, dtype=int64)
    imax = ones( nranks, dtype=int64) * (len(arr) - 1)
    done = zeros(nranks, dtype=bool)

    local_median = zeros(nranks, dtype=arr.dtype)
    weight       = zeros(nranks, dtype=float)
    est_median   = zeros(nranks, dtype=arr.dtype)

    # Arrays with result
    res_median   = zeros(nranks, dtype=arr.dtype)
    res_min_rank = zeros(nranks, dtype=int64)
    res_max_rank = zeros(nranks, dtype=int64)

    # Iterate until we find all splitting points
    while any(done==False):

        # Estimate median of elements in active ranges for each rank
        nlocal  = maximum(0,imax-imin+1  )              # Array with local size of active range for each rank
        nactive     = gather_to_2d_array(nlocal, comm)
        nactive_sum = sum(nactive, axis=0)              # sum over tasks, gives one element per rank
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
        ifirst = searchsorted(arr, est_median, side='left')
        ilast  = searchsorted(arr, est_median, side='right')
        ifirst_all = gather_to_2d_array(ifirst, comm) # Returns 2D array of size [comm_size, nranks]
        ilast_all  = gather_to_2d_array(ilast, comm)
        fsum = sum(ifirst_all, axis=0, dtype=int64) # Sum over tasks, leaves one element per rank to find
        lsum = sum(ilast_all,  axis=0, dtype=int64)

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
    if comm is None:
        comm = MPI.COMM_WORLD

    # mpi4py doesn't like wrong endian data!
    if not(arr.dtype.isnative):
        print "parallel_sort.py: Unable to operate on non-native endian data!"
        comm.Abort()

    # Record starting time
    if verbose:
        comm.Barrier()
        t0 = time.time()

    # Sanity check input
    if not(hasattr(arr,"dtype")) or not(hasattr(arr,"shape")):
        print "Can only sort arrays!"
        comm.Abort()
    if len(arr.shape) != 1:
        print "Can only sort 1D data!"
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
        ntot        = sum(nperproc)

        # Sort array locally
        if verbose and mycomm_rank==0:
            print "Sorting local elements, t = ", time.time()-t0
        sort_idx1 = my_argsort(arr)
        arr[:] = arr[sort_idx1]
        if not(return_index):
            del sort_idx1

        # Find global rank of first value to put on each processor
        split_rank = cumsum(nperproc) - nperproc

        # Find values at which we want to split the sorted
        # array and also minimum and maximum global ranks
        # of these values (first element on processor mycomm_rank=0 has
        # global rank 0)
        if verbose and mycomm_rank==0:
            print "Calculating splitting points, t = ", time.time()-t0

        # Find the values with ranks in split_rank, and also the maximum and
        # minimum ranks which have this value
        val, min_rank, max_rank = find_splitting_points(arr, split_rank, mycomm)
        split_instance = split_rank - min_rank

        # Find out how many instances of each splitting value we have
        # on each processor
        if arr.shape[0] > 0:
            first_ind  = searchsorted(arr, val, side='left')
            last_ind   = searchsorted(arr, val, side='right') - 1
            nval_local = last_ind - first_ind + 1
            first_ind[first_ind >= len(arr)] = len(arr) - 1
            nval_local[arr[first_ind] != val] = 0 
        else:
            nval_local = zeros(len(val), dtype=index_dtype)
        nval_local = asarray(nval_local, dtype=index_dtype)

        # Find out how many instances of each value there are on each
        # lower ranked processor - indexes are (iproc, ival) after gather.
        nval_all = ndarray((mycomm_size, mycomm_size), dtype=index_dtype)
        mycomm.Allgather(nval_local, nval_all) 
        nval_lower = zeros(len(val), dtype=index_dtype)
        for ival in range(len(val)):
            if mycomm_rank > 0:
                nval_lower[ival] = sum(nval_all[0:mycomm_rank,ival])
            else:
                nval_lower[ival] = 0

        # Determine which local elements are to go to which other
        # processor
        if verbose and mycomm_rank==0:
            print "Determining destination for each element, t = ", time.time()-t0
        first_to_send = 0
        send_count = zeros(mycomm_size, dtype=index_dtype)

        # Find array indexes of splitting point values
        # It's much faster to do this in one pass here than to do
        # each one separately inside the loop below.
        ranks = arange(mycomm_size, dtype=int32)
        val_first_index_arr = searchsorted(arr, val[ranks], side='left')
        val_last_index_arr  = searchsorted(arr, val[ranks], side='right') - 1

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
                    if arr[val_first_index] > val[rank+1]:
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
        send_displ = cumsum(send_count) - send_count
        recv_count = ndarray(mycomm_size, dtype=index_dtype)
        mycomm.Alltoall(send_count, recv_count)
        recv_displ = cumsum(recv_count) - recv_count

        # Exchange data
        if verbose and mycomm_rank==0:
            print "Exchanging data, t = ", time.time()-t0
        arr_tmp = ndarray(sum(recv_count), dtype=arr.dtype)
        my_alltoallv(arr,     send_count, send_displ,
                     arr_tmp, recv_count, recv_displ,
                     comm=mycomm)

        # Sort local data
        if verbose and mycomm_rank==0:
            print "Sorting local elements, t = ", time.time()-t0
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
                print "Making index array, t = ", time.time()-t0
            # Rearrange indices using index from initial local sort
            # of the data array
            index = arange(arr.shape[0], dtype=index_dtype)
            if mycomm_rank > 0:
                index[:] += sum(nperproc[0:mycomm_rank], dtype=index_dtype)
            index = index[sort_idx1]
            del sort_idx1
            # Exchange index data in same way as array values
            index_tmp = ndarray(sum(recv_count), dtype=index_dtype)
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
            index = ndarray(0, dtype=index_dtype)

    if verbose and mycomm_rank==0:
        print "Parallel sort finished, t = ", time.time()-t0

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
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    # mpi4py doesn't like wrong endian data!
    if not(arr1.dtype.isnative) or not(arr2.dtype.isnative):
        print "parallel_sort.py: Unable to operate on non-native endian data!"
        comm.Abort()

    # Sanity checks on input arrays
    if len(arr1.shape) != 1 or len(arr2.shape) != 1:
        print "Can only match elements between 1D arrays!"
        comm.Abort()

    # If arr1 has no elements we just return an empty index array
    if comm.allreduce(arr1.shape[0]) == 0:
        return ndarray(0, dtype=index_dtype)

    # If arr2 has no elements then nothing will match
    if comm.allreduce(arr2.shape[0]) == 0:
        return -ones(arr1.shape, dtype=index_dtype)

    # Make a sorted copy of arr2 if necessary
    if not(arr2_sorted):
        arr2_ordered = arr2.copy()
        sort_arr2 = parallel_sort(arr2_ordered, return_index=True, comm=comm)
    else:
        arr2_ordered = arr2

    # Make array of initial indexes of arr2 elements and arrange in sorted order
    n_per_task = asarray(comm.allgather(arr2.shape[0]), dtype=index_dtype)
    first_on_task = cumsum(n_per_task) - n_per_task
    index_in = arange(arr2.shape[0], dtype=index_dtype) + first_on_task[comm_rank]
    if not(arr2_sorted):
        index_in = fetch_elements(index_in, sort_arr2, comm=comm)
        del sort_arr2

    # Find range of arr2 values on each task after sorting
    min_on_task = ndarray(comm_size, dtype=arr2.dtype)
    max_on_task = ndarray(comm_size, dtype=arr2.dtype)
    if arr2_ordered.shape[0] > 0:
        min_val = amin(arr2_ordered)
        max_val = amax(arr2_ordered)
    else:
        # If there are no values set min and max to zero
        min_val = asarray(0, dtype=arr2_ordered.dtype)
        max_val = asarray(0, dtype=arr2_ordered.dtype)
    comm.Allgather(min_val, min_on_task)
    comm.Allgather(max_val, max_on_task)
    # Record which tasks have >0 arr2 elements
    have_arr2 = asarray(comm.allgather(arr2_ordered.shape[0] > 0), dtype=bool)

    # Sort local arr1 values
    idx = my_argsort(arr1)
    arr1_ls = arr1[idx]
    
    # Decide which elements of arr1 to send to which tasks
    # Handle duplicate values in arr2 by sending to lowest numbered task.
    # First we calculate offsets and counts for tasks with >0 elements in arr2.
    send_displ      = searchsorted(arr1_ls, min_on_task[have_arr2], side="left")
    send_displ_next = searchsorted(arr1_ls, max_on_task[have_arr2], side="right")
    if sum(have_arr2) > 1:
        send_displ[1:] = send_displ_next[:-1]
    send_count = send_displ_next - send_displ
    send_count[send_count<0] = 0
    assert sum(send_count) <= arr1_ls.shape[0]
    assert all(send_displ >= 0)
    assert all(send_displ+send_count <= arr1_ls.shape[0])

    # Expand displacement and count arrays to include tasks with no arr2 elements
    send_displ_all = zeros(comm_size, dtype=send_displ.dtype)
    send_displ_all[have_arr2] = send_displ
    send_count_all = zeros(comm_size, dtype=send_count.dtype)
    send_count_all[have_arr2] = send_count # Needs to be zero where have_arr2 is false
    send_count = send_count_all
    send_displ = send_displ_all

    # Transfer the data
    recv_count = ndarray(comm_size, dtype=index_dtype)
    comm.Alltoall(send_count, recv_count)
    recv_displ = cumsum(recv_count) - recv_count
    arr1_recv = ndarray(sum(recv_count), dtype=arr1.dtype)
    my_alltoallv(arr1_ls,   send_count, send_displ,
                 arr1_recv, recv_count, recv_displ,
                 comm=comm)
    del arr1_ls

    # For each imported arr1 element, find global rank of matching arr2 element
    ptr = searchsorted(arr2_ordered, arr1_recv, side="left")
    ptr[ptr<0] = 0
    ptr[ptr>=arr2_ordered.shape[0]] = 0 
    ptr[arr2_ordered[ptr] != arr1_recv] = -1
    del arr1_recv
    del arr2_ordered
    index_return = -ones(ptr.shape, dtype=index_dtype)
    index_return[ptr>=0] = index_in[ptr[ptr>=0]]
    del index_in
    del ptr

    # Return the index info
    index_out = zeros(arr1.shape, dtype=index_dtype) - 1
    my_alltoallv(index_return, recv_count, recv_displ,
                 index_out,    send_count, send_displ,
                 comm=comm)
    del index_return

    # Restore original order
    index = empty_like(index_out)
    index[idx] = index_out
    del index_out
    del idx

    return index


    
def small_test():
    """Test sorting code on random arrays"""

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    for i in range(2000):

        if comm_rank == 0:
            print "Test ", i

        # Make random test aray
        n   = random.randint(200) + 0
        arr = random.randint(10, size=n)

        # Write out input
        for rank in range(comm_size):
            if comm_rank == rank:
                print comm_rank,":",arr
                sys.stdout.flush()
            comm.Barrier()

        # Keep a copy of the original
        orig = arr.copy()

        # Sort
        if comm_rank == 0:
            print "start sorting"
        index = parallel_sort(arr, return_index=True)
        if comm_rank == 0:
            print "done sorting"
            
        # Write out results
        for rank in range(comm_size):
            if comm_rank == rank:
                print comm_rank,":",arr
                sys.stdout.flush()
            comm.Barrier()

        # Verify order locally
        arr_sorted = arr
        delta = arr_sorted[1:] - arr_sorted[:-1]
        if any(delta<0.0):
            print "Local values are not sorted correctly!"
            comm.Abort()

        # Check ordering between processors
        if n > 0:
            local_min = amin(arr_sorted)
            local_max = amax(arr_sorted)
        else:
            local_min = 0
            local_max = 0
        all_min = comm.allgather(local_min)
        all_max = comm.allgather(local_max)
        all_n   = comm.allgather(n)
        ind = (all_n > 0)
        if sum(ind) > 1:
            all_min = all_min[ind]
            all_max = all_max[ind]
            for rank in range(1, sum(ind)):
                if all_min[rank] < all_max[rank-1]:
                    print "Values are not sorted correctly between processors!"
                    comm.Abort()

        # Check that we can reconstruct the array using the index
        arr_from_index = fetch_elements(orig, index)
        if any(arr_from_index != arr):
            print "Index doesn't work!"
            print comm_rank,"-", orig, arr, arr_from_index, index
            comm.Abort()
        else:
            if comm_rank==0:
                print "Index is ok"

        comm.Barrier()
        if comm_rank == 0:
            print "Array is sorted correctly"


  
def big_test():
    """Try sorting lots of elements!"""

    nmax = int(sys.argv[1])

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Test parallel sort routine - first generate test dataset
    random.seed(35915)
    n   = int((random.rand() + 0.1) * nmax)
    arr = random.rand(n)
    ntot1 = comm.allreduce(arr.shape[0])

    if comm_rank == 0:
        print "Total elements = ", ntot1

    # Sort the array
    comm.Barrier()
    t0 = time.time()
    index = parallel_sort(arr, return_index=True, verbose=True)
    comm.Barrier()
    t1 = time.time()
    if comm_rank==0:
        print "Elapsed time (s) = ", t1-t0

    # Verify order locally
    arr_sorted = arr
    delta = arr_sorted[1:] - arr_sorted[:-1]
    if any(delta<0.0):
        print "Local values are not sorted correctly!"
        comm.Abort()

    # Check ordering between processors
    if n > 0:
        local_min = amin(arr_sorted)
        local_max = amax(arr_sorted)
    else:
        local_min = 0
        local_max = 0
    all_min = comm.allgather(local_min)
    all_max = comm.allgather(local_max)
    all_n   = comm.allgather(n)
    ind = (all_n > 0)
    if sum(ind) > 1:
        all_min = all_min[ind]
        all_max = all_max[ind]
        for rank in range(1, sum(ind)):
            if all_min[rank] < all_max[rank-1]:
                print "Values are not sorted correctly between processors!"
                comm.Abort()

    # If we get here, output array is sorted
    comm.Barrier()
    if comm_rank == 0:
        print "Array is sorted correctly"


def repartition_test():

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    for i in range(10):

        if comm_rank == 0:
            print "Test ", i

        # Make random test aray
        n   = random.randint(5) + 0
        arr = random.randint(10, size=n)    

        # Write out input
        for rank in range(comm_size):
            if comm_rank == rank:
                print comm_rank," - input:",arr
                sys.stdout.flush()
            comm.Barrier()        

        # Pick random destination for each element
        dest   = random.randint(comm_size, size=n)

        # Count elements we want on each processor
        ndesired = zeros(comm_size, dtype=int)
        for rank in range(comm_size):
            nlocal = sum(dest==rank)
            ndesired[rank] = comm.allreduce(nlocal)

        print comm_rank, "- Ndesired = ", ndesired
        sys.stdout.flush()
        comm.Barrier()

        # Repartition
        new_arr = repartition(arr, ndesired)

        # Write out output
        for rank in range(comm_size):
            if comm_rank == rank:
                print comm_rank," - output:",new_arr
                sys.stdout.flush()
            comm.Barrier()


if __name__ == "__main__":
    big_test()
