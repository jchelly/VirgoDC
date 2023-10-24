#!/bin/env python

import numpy as np

import virgo.mpi.parallel_sort as psort


def match_halos(grnr1, ids1, grnr2, ids2, local_nr_halos, comm=None):
    """
    Find matching halos between two snapshots by looking for particle IDs
    in common.

    grnr1 and ids1 contain the group membership and particle ID for all
    particles in halos in snapshot 1.

    grnr2 and ids2 contain the group membership and particle ID for all
    particles in halos in snapshot 2.

    If there are N halos, group membership goes from 0 to N-1, with -1
    indicating that a particle is in no halo. For each halo in snapshot
    1 we compute the number of particles which go to each halo in snapshot 2
    and find the halo which received the largest number.
    
    Returns a distributed array which has one element for each halo in snap1
    which contains the index of the matched halo in snap2, or -1 if there is
    no match.

    local_nr_halos determines how the output is partitioned between ranks.
    It contains the number of halos to be stored on this rank.
    """

    # Get communicator to use
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    # For each particle ID in snap1, find the global index of the same ID in snap2
    snap2_id_index = psort.parallel_match(ids1, ids2, comm=comm)

    # Construct an array to store pairs of group indexes
    grnr_pair_t = np.dtype([("grnr1", grnr1.dtype),
                            ("grnr2", grnr2.dtype)])
    grnr_pair = np.ndarray(len(ids1), dtype=grnr_pair_t)
    grnr_pair["grnr1"] = grnr1
    grnr_pair["grnr2"] = -1

    # For each particle ID in snap1, fetch the group membership of the same particle in snap2
    matched = snap2_id_index >= 0
    grnr_pair["grnr2"][matched] = psort.fetch_elements(grnr2, snap2_id_index[matched], comm=comm)
    del snap2_id_index
    del matched

    # Only keep entries where the particle is in halos at both snapshots
    keep = (grnr_pair["grnr1"] >= 0) & (grnr_pair["grnr2"] >= 0)
    grnr_pair = grnr_pair[keep]
    del keep
        
    # Find unique (grnr1, grnr2) combinations and number of instances of each
    unique_grnr, unique_grnr_count = psort.parallel_unique(grnr_pair, comm=comm, arr_sorted=False,
                                                           return_counts=True, repartition_output=True)

    # Compute destination rank for each pair:
    # For simplicity we put equal numbers of halos on each MPI rank except that
    # the last rank gets any extra.
    total_nr_halos = comm.allreduce(local_nr_halos)
    halos_per_rank = total_nr_halos // comm_size
    destination = np.clip(unique_grnr["grnr1"] // halos_per_rank, a_min=None, a_max=comm_size-1)
    local_size = halos_per_rank
    if comm_rank == comm_size - 1:
        local_size += total_nr_halos % comm_size
    assert comm.allreduce(local_size) == total_nr_halos
    local_offset = comm_rank * halos_per_rank
    
    # Compute number of pairs to send to each rank
    send_count = np.bincount(destination, minlength=comm_size)
    send_offset = np.cumsum(send_count) - send_count
    recv_count = np.empty_like(send_count)
    comm.Alltoall(send_count, recv_count)
    recv_offset = np.cumsum(recv_count) - recv_count

    # Exchange pairs
    total_nr_recv = np.sum(recv_count)
    unique_grnr_recv = np.ndarray(total_nr_recv, dtype=unique_grnr.dtype)
    psort.my_alltoallv(unique_grnr, send_count, send_offset,
                       unique_grnr_recv, recv_count, recv_offset,
                       comm=comm)
    del unique_grnr
    
    # Exchange pair counts
    unique_grnr_count_recv = np.ndarray(total_nr_recv, dtype=unique_grnr_count.dtype)
    psort.my_alltoallv(unique_grnr_count, send_count, send_offset,
                       unique_grnr_count_recv, recv_count, recv_offset,
                       comm=comm)
    del unique_grnr_count

    # Allocate output array
    matched_grnr2 = -np.ones(local_size, dtype=unique_grnr_recv["grnr2"].dtype)
    matched_count = np.zeros(local_size, dtype=int)

    # Loop over received pairs
    grnr1 = unique_grnr_recv["grnr1"]
    grnr2 = unique_grnr_recv["grnr2"]
    for recv_nr in range(total_nr_recv):

        # Get the local index of the halo at snap1
        local_halo_nr = grnr1[recv_nr] - local_offset
        assert local_halo_nr >= 0
        assert local_halo_nr < local_size

        # Check if the received count is higher than the highest so far
        if unique_grnr_count_recv[recv_nr] > matched_count[local_halo_nr]:
            # We have a new highest count so far
            matched_grnr2[local_halo_nr] = grnr2[recv_nr]
            matched_count[local_halo_nr] = unique_grnr_count_recv[recv_nr]
        elif unique_grnr_count_recv[recv_nr] == matched_count[local_halo_nr]:
            # In case of a tie go for lowest group number
            if(grnr2[recv_nr] <  matched_grnr2[local_halo_nr]):
                matched_grnr2[local_halo_nr] = grnr2[recv_nr]
                matched_count[local_halo_nr] = unique_grnr_count_recv[recv_nr]
                
    # Repartition and return the output array
    ndesired = np.asarray(comm.allgather(local_nr_halos))
    return psort.repartition(matched_grnr2, ndesired=ndesired, comm=comm)


def consistent_match(match_index_12, match_index_21):
    """
    For each halo in catalogue 1, determine if its match in catalogue 2
    points back at it.

    match_index_12 has one entry for each halo in catalogue 1 and
    specifies the matching halo in catalogue 2 (or -1 for not matched)

    match_index_21 has one entry for each halo in catalogue 2 and
    specifies the matching halo in catalogue 1 (or -1 for not matched)

    Returns an array with 1 for a match and 0 otherwise.
    """

    # Find the global array indexes of halos stored on this rank
    nr_local_halos = len(match_index_12)
    local_halo_offset = comm.scan(nr_local_halos) - nr_local_halos
    local_halo_index = np.arange(
        local_halo_offset, local_halo_offset + nr_local_halos, dtype=int
    )

    # For each halo, find the halo that its match in the other catalogue was matched with
    match_back = -np.ones(nr_local_halos, dtype=int)
    has_match = match_index_12 >= 0
    match_back[has_match] = psort.fetch_elements(
        match_index_21, match_index_12[has_match], comm=comm
    )

    # If we retrieved our own halo index, we have a match
    return np.where(match_back == local_halo_index, 1, 0)
