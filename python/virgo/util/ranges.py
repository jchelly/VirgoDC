#!/bin/env python

import numpy as np

def assign_ranges(array, offset, length, value):
    """
    For each (o, l, v) in zip(offset, length, value) set
    array[o:o+l] = v.

    Slices [o:o+l] must be in sorted order and must not
    overlap.

    Modifies input array in place. If array and values
    are multidimensional, offsets and lengths are assumed
    to refer to the first dimension.
    """

    # Ensure input is an array
    array = np.asarray(array)

    # Sanity check ranges
    offset = np.asarray(offset).astype(np.intp)
    length = np.asarray(length).astype(np.intp)
    value  = np.asarray(value)
    if len(offset.shape)!=1 or len(length.shape)!=1:
        raise ValueError("Length and offset arrays must be 1D")
    if not(offset.shape[0] == length.shape[0] == value.shape[0]):
        raise ValueError("Length, offset and value arrays must have same first dimension")
    nrange = offset.shape[0]

    if nrange == 0:
        # If we have zero ranges, there's nothing to do
        return
    elif nrange == 1:
        # Only one range, can just assign it
        array[offset[0]:offset[0]+length[0],...] = value[0,...]
    else:
        # Have two or more ranges to assign.
        # Calculate sizes of gaps between ranges - i.e. elements not to be assigned
        gap_length = np.zeros(nrange+1, dtype=np.intp)
        gap_length[0]    = offset[0] # Gap before first range
        gap_length[-1]   = array.shape[0] - (offset[-1]+length[-1]) # Gap after last range
        gap_length[1:-1] = offset[1:] - (offset[:-1] + length[:-1]) # Gaps between ranges

        # Ranges plus gaps should account for every element in array exactly once
        if np.sum(length) + np.sum(gap_length) != array.shape[0]:
            raise ValueError("Something wrong here - maybe overlapping ranges?")

        # Make boolean mask to indicate values in array which will be updated
        all_lengths = np.zeros(gap_length.shape[0]+length.shape[0], dtype=np.intp)
        all_lengths[0::2] = gap_length
        all_lengths[1::2] = length
        mask = np.zeros(gap_length.shape[0]+length.shape[0], dtype=np.bool)
        mask[1::2] = True
        mask = np.repeat(mask, all_lengths)

        # Carry out assignment
        array[mask,...] = np.repeat(value, length, axis=0)


def sum_ranges(array, offset, length, dtype=None, weight=None,
               normalize=True):
    """
    Calculate sum of elements of array in the ranges
    given by offset and length arrays. I.e. result is

    [sum(array[offset[0]:offset[0]+length[0]]),
     sum(array[offset[1]:offset[1]+length[1]]),
     ...
     sum(array[offset[-1]:offset[-1]+length[-1]])]

    If array and values are multidimensional, offsets and
    lengths are assumed to refer to the first dimension and
    sum is done along the first dimension.

    Ranges to sum must be in sorted order and not overlap.

    If weight is not None then we do a weighted sum. Weight
    should be a 1D array of length array.shape[0]. Result
    for each range is divided by sum of weights in that range
    if normalize=True.
    """
    
    # Ensure input is an array
    array = np.asarray(array)
    if dtype is None:
        dtype = array.dtype
    offset = np.asarray(offset).astype(np.intp)
    length = np.asarray(length).astype(np.intp)

    # Create mask to pick out elements to sum
    mask = np.zeros(array.shape[0], dtype=np.bool)
    assign_ranges(mask, offset, length, np.ones(offset.shape[0], dtype=np.bool))
    
    if weight is None:
        # No weight, so just return sum of ranges
        return np.add.reduceat(array[mask,...], 
                               np.cumsum(length)-length,
                               axis=0, dtype=dtype)
    else:
        # Weighted sum.
        a_x_w     = (array[mask,...].T*weight[mask]).T # transpose so we can broadcast weight
        sum_a_x_w = np.add.reduceat(a_x_w, np.cumsum(length)-length,
                                    axis=0, dtype=dtype)
        if normalize:
            sum_w     = np.add.reduceat(weight[mask], np.cumsum(length)-length,
                                        axis=0, dtype=dtype)
            return (sum_a_x_w.T / sum_w).T
        else:
            return sum_a_x_w
