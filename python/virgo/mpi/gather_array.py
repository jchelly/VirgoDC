#!/bin/env python

import numpy as np
from mpi4py import MPI


def allgather_array(data, comm=None, axis=0):
    """
    To do an MPI_ALLGATHERV, we just set the root parameter
    to None.
    """
    return gather_vector(data, root=None, comm=comm, axis=axis)


def gather_array(data, root=0, comm=None, axis=0):
    """
    This does an MPI_GATHERV, handling the calculation of
    count and offset parameters by assuming that the output
    should be contiguous and in order of source task.

    If root=None we do an MPI_ALLGATHERV - all tasks receive the
    result. Otherwise tasks other than root return None.

    Multidimensional arrays are gathered along the axis specified
    by the axis parameter. This routine is most efficient if
    axis=0. If axis!=0 the input array will have to be copied
    and the result will likely be a view of an array with
    the axes rearranged so that the specified axis is first.
    """

    # Get communicator to use
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Make chosen axis the first index
    if axis != 0:
        data = data.swapaxes(0, axis)

    # Ensure array is contiguous
    data = np.ascontiguousarray(data)

    # Find total number of elements
    ntot = comm.allreduce(data.shape[0])

    # Find number of elements per unit along the first axis
    nelements = 1
    for s in data.shape[1:]:
        nelements *= s

    # Determine shape and type of output dataset
    out_dtype    = data.dtype
    out_shape    = list(data.shape)
    out_shape[0] = ntot

    # Calculate counts and displacements
    counts  = np.asarray(comm.allgather(data.shape[0]), dtype=int)
    displ   = np.cumsum(counts) - counts
    counts *= nelements
    displ  *= nelements

    # Allocate output array on root rank
    if comm_rank == root or root is None:
        outdata = np.ndarray(out_shape, dtype=out_dtype)
    else:
        outdata = None

    # Transfer data
    if root is None:
        comm.Allgatherv(data, (outdata, (counts, displ)))
    else:
        comm.Gatherv(data, (outdata, (counts, displ)), root=root)

    # Swap axes back if necessary
    if axis != 0 and outdata is not None:
        outdata = outdata.swapaxes(0, axis)

    # Return result
    return outdata




if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    data = np.arange(3) + 10*comm_rank
    data = np.asarray((data,data))
    
    for i in range(comm_size):
        if comm_rank == i:
            print "Data on task ", comm_rank, ":"
            print data, data.shape
        comm.barrier()

    result = gather_array(data, axis=1)
    if result is not None:
        print "Result is ", result, result.shape
