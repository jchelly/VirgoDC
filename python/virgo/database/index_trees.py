#!/bin/env python
#
# Routines for generating depth first indexes for Eagle merger trees
#

import numpy as np
from virgo.util.match import match


def depth_first_index(nodeIndex, descendantIndex, mBranch):
    """
    Calculate depth first indexing for the supplied tree(s).
    Progenitors are ordered by descending mBranch.

    Returns:

    depthFirst     : depth first index of each object
    endMainBranch  : depthFirst at end of main branch
    lastProgenitor : depthFirst of last progenitor
    """

    # Find array index of descendant for each halo
    desc_ptr = match(descendantIndex, nodeIndex)

    # Allocate arrays to store linked lists of progenitors
    # (which will be sorted into descending order of mBranch)
    prog_ptr = -np.ones_like(desc_ptr) # points to first progenitor
    next_ptr = -np.ones_like(desc_ptr) # points to next progenitor of same descendant

    # Loop over all possible progenitor halos iprog in ascending
    # order of mBranch
    for iprog in np.argsort(mBranch):
        idesc = desc_ptr[iprog]
        if idesc >= 0:
            # Halo idesc is descendant of iprog. 
            # Add it to the start of idesc's linked list.
            # Since we consider halos in ascending order of mBranch,
            # the resulting list will be in *descending* order of mBranch. 
            if prog_ptr[idesc] == -1:
                # This is the first progenitor we found for halo i
                prog_ptr[idesc] = iprog
            else:
                # Replace current first progenitor with this one
                itmp = prog_ptr[idesc]
                prog_ptr[idesc] = iprog
                next_ptr[iprog] = itmp
                
    # Alloc. arrays for depth first indexes
    depthFirst = -np.ones_like(descendantIndex)
                
    # Loop over final halos
    next_index = 1
    ntot = nodeIndex.shape[0]
    for ifinal in np.where(desc_ptr<0)[0]:

        # Walk the tree
        ihalo = ifinal
        next_prog = [-1] # Final halo has no siblings
        while True:

            # Assign the next depth first ID to this halo
            depthFirst[ihalo] = next_index
            next_index += 1

            # Go to first progenitor while there is one, assigning
            # indexes along the way
            while prog_ptr[ihalo] >= 0:

                ihalo = prog_ptr[ihalo]           # go to first progenitor
                next_prog.append(next_ptr[ihalo]) # push sibling index to stack
                depthFirst[ihalo] = next_index    # assign next index
                next_index += 1
                
            # Now we're at the end of the branch, go back up
            # until we find a sibling halo
            while desc_ptr[ihalo] >= 0:
                isibling = next_prog.pop()
                if isibling >= 0:
                    # This one has a sibling, so visit it
                    ihalo = isibling
                    # Push sibling's sibling (if any) to the stack
                    next_prog.append(next_ptr[ihalo])
                    break
                else:
                    # No sibling, go to descendant
                    ihalo = desc_ptr[ihalo]

            # If we get back to where we started, we're done
            if ihalo == ifinal:
                break

    # Alloc. arrays for last progenitor and end of main branch
    endMainBranch  = depthFirst.copy()
    lastProgenitor = depthFirst.copy()

    # Loop over halos in descending order of depth first ID
    for iprog in np.argsort(-depthFirst):
        idesc = desc_ptr[iprog]
        if idesc >= 0:
            # Update maximum ID of any progenitor of this descendant
            lastProgenitor[idesc] = max(lastProgenitor[iprog], lastProgenitor[idesc])
            # If we're on the main branch, updated end of main branch ID
            if prog_ptr[idesc] == iprog:
                endMainBranch[idesc] = max(endMainBranch[iprog], endMainBranch[idesc])

    # All array elements should have been set
    assert np.all(endMainBranch  >= 0)
    assert np.all(lastProgenitor >= 0)
    assert np.all(depthFirst     >= 0)
    
    # Should always have lastProgenitor >= endMainBranch >= depthFirst
    assert np.all(lastProgenitor >= endMainBranch)
    assert np.all(endMainBranch  >= depthFirst)

    # Return array with the indexes
    return depthFirst, endMainBranch, lastProgenitor


def find_main_progenitor(nodeIndex, descendantIndex, snapnum, mbranch):
    """
    Identify main progenitor using branch mass
    Returns number of ties (equal max mbranches) and
    main progenitor ID.
    """

    # Get number of nodes
    n = nodeIndex.shape[0]

    # Output array
    mainprog = -np.ones_like(nodeIndex)
    
    # Get order in which to do comparisons
    idx  = np.lexsort((-nodeIndex,snapnum))

    # Get descendant array index for each node
    ptr = match(descendantIndex, nodeIndex)
 
    # Find highest main branch mass for each node
    mbranch_max = np.zeros_like(mbranch)
    min_prog_snap = np.zeros_like(mbranch, dtype=np.int32) + 100
    for i in idx:
        j = ptr[i]
        if j >= 0:
            if mbranch[i] > mbranch_max[j]:
                mbranch_max[j] = mbranch[i]
                mainprog[j] = nodeIndex[i]
                min_prog_snap[j] = snapnum[i]

    # Check for ties
    eps = 1.0e-5
    ntie = np.zeros(n, dtype=np.int32)
    for i in idx:
        j = ptr[i]
        if j >= 0:
            if abs(mbranch[i]-mbranch_max[j])/mbranch[i] < eps:
                ntie[j] += 1
    ntie -= 1
    ntie[ntie<0] = 0

    return ntie, mainprog


def find_progenitor_tree_mass(nodeIndex, descendantIndex, snapnum, mass):
    """
    For each object, return total mass in past merger tree.
    This includes the object's own mass.
    """

    # Get number of nodes
    n = nodeIndex.shape[0]

    # Sort into ascending order of snapnum.
    idx  = np.lexsort((-nodeIndex,snapnum))
    ni   = nodeIndex[idx]
    di   = descendantIndex[idx]
    snap = snapnum[idx]
    mbranch = mass[idx].astype(np.float64) # will eventually contain total main branch mass

    # Identify descendants
    ptr = match(di, ni)
    
    # Store descendant's mass for each object
    mdesc = np.zeros(n, dtype=mass.dtype)
    mdesc[ptr>=0] = mbranch[ptr[ptr>=0]]
    
    # Loop over objects in order of increasing snapnum
    for i in xrange(n):
        # Check if we have a descendant
        if ptr[i] >= 0:
            # Accumulate descendant's total progenitor mass
            mbranch[ptr[i]] += mbranch[i]

    # Put branch mass back into same order as input
    mbranch_out = np.empty_like(mass)
    mbranch_out[idx] = mbranch

    return mbranch_out


def find_progenitor_branch_mass_delucia(nodeIndex, descendantIndex, snapnum, mass):
    """
    Identify main progenitor branch mass of each object using algorithm
    of DeLucia and Blaizot (2007) (see equation 1 in the paper).

    Returns main branch mass
    """

    # Get number of nodes
    n = nodeIndex.shape[0]

    # Sort into ascending order of snapnum.
    idx  = np.lexsort((-nodeIndex,snapnum))
    ni   = nodeIndex[idx]
    di   = descendantIndex[idx]
    snap = snapnum[idx]
    mbranch = mass[idx] # will eventually contain total main branch mass

    # Identify descendants
    ptr = match(di, ni)
    
    # Store descendant's mass for each object
    mdesc = np.zeros(n, dtype=mass.dtype)
    mdesc[ptr>=0] = mbranch[ptr[ptr>=0]]
    
    # Loop over objects in order of increasing snapnum
    for i in xrange(n):
        # Check if we have a descendant
        if ptr[i] >= 0:
            # If our branch mass plus the descendant mass is greater than
            # the descendants current branch mass, replace the descendant's
            # branch mass.
            desc_mbranch = mbranch[ptr[i]]
            prog_mbranch = mdesc[i]+mbranch[i]
            if prog_mbranch > desc_mbranch:
                mbranch[ptr[i]] = prog_mbranch

    # Put branch mass back into same order as input
    mbranch_out = np.empty_like(mbranch)
    mbranch_out[idx] = mbranch

    return mbranch_out




