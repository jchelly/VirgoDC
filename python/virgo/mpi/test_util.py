#!/bin/env python

import numpy as np
import pytest
import virgo.mpi.util as util

def make_random_lengths_and_offsets(min_size, max_size, nr_groups):
    """
    Generate a distributed, random group catalogue to test the
    group_index_from_length_and_offset() function.
    """

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if comm_rank == 0:
    
        # Generate the lengths and offsets on rank 0
        lengths = np.random.randint(low=min_size, high=max_size, size=nr_groups)
        offsets = np.cumsum(lengths) - lengths
        total_nr_ids = np.sum(lengths)

        # Decide how to distribute lengths and offsets between MPI ranks
        group_rank = np.random.randint(comm_size, size=len(lengths))
        nr_per_rank = np.bincount(group_rank, minlength=comm_size)
        offset_for_rank = np.cumsum(nr_per_rank) - nr_per_rank

        # Distribute lengths and offsets
        for rank_nr in range(comm_size):
            i1 = offset_for_rank[rank_nr]
            i2 = i1 + nr_per_rank[rank_nr]
            if rank_nr == 0:
                local_lengths = lengths[i1:i2]
                local_offsets = offsets[i1:i2]
            else:
                data = (lengths[i1:i2], offsets[i1:i2])
                comm.send(data, dest=rank_nr)
    else:
        # Receive lengths and offsets from rank zero
        local_lengths, local_offsets = comm.recv(source=0)
        lengths = None
        offsets = None
        total_nr_ids = None

    # Decide how to split the IDs between ranks
    if comm_rank == 0:
        rank_for_id = np.random.randint(comm_size, size=total_nr_ids)
        ids_per_rank = np.bincount(rank_for_id, minlength=total_nr_ids)
    else:
        ids_per_rank = None
    ids_per_rank = comm.bcast(ids_per_rank)
    nr_local_ids = ids_per_rank[comm_rank]

    # Compute group index for each particle
    if comm_rank == 0:
        grnr = -np.ones(total_nr_ids, dtype=int)
        for i, (l, o) in enumerate(zip(lengths, offsets)):
            grnr[o:o+l] = i
    else:
        grnr = None

    return local_lengths, local_offsets, nr_local_ids, lengths, offsets, total_nr_ids, grnr


def check_lengths_offsets(min_size, max_size, nr_groups):
    """
    Make a randomly generated group catalogue with known group numbers
    and try to reproduce the group numbers with 
    group_index_from_length_and_offset().
    """

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    # Create the test dataset
    (local_lengths, local_offsets,
     nr_local_ids, lengths, offsets,
     total_nr_ids, grnr) = make_random_lengths_and_offsets(min_size, max_size, nr_groups)

    # Try to recompute the group indexes
    grnr_local = util.group_index_from_length_and_offset(local_lengths, local_offsets, nr_local_ids)

    # Gather result on rank 0 and check
    grnr_gathered = comm.gather(grnr_local)
    if comm_rank == 0:
        grnr_gathered = np.concatenate(grnr_gathered)
        all_equal = np.all(grnr_gathered == grnr)
    else:
        all_equal = None
    all_equal = comm.bcast(all_equal)

    assert all_equal, "Computed group numbers are not correct"

@pytest.mark.mpi
def test_lengths_offsets_small():
    check_lengths_offsets(min_size=10, max_size=20, nr_groups=10000)

@pytest.mark.mpi
def test_lengths_offsets_large():
    check_lengths_offsets(min_size=1000, max_size=100000, nr_groups=100)

@pytest.mark.mpi
def test_lengths_offsets_range():
    check_lengths_offsets(min_size=10, max_size=100000, nr_groups=1000)
    
