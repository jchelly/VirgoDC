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
               normalize=False):
    """
    Calculate sum of elements of array in the ranges
    given by offset and length arrays. I.e. result is

    [sum(array[offset[0]:offset[0]+length[0]]),
     sum(array[offset[1]:offset[1]+length[1]]),
     ...
     sum(array[offset[-1]:offset[-1]+length[-1]])]

    If array is multidimensional, offsets and lengths are assumed
    to refer to the first dimension and sum is done along the 
    first dimension.

    If dtype is not None then the calculation is carried out
    using the specified type and the result will by of type
    dtype. Otherwise array.dtype is used.

    Ranges to sum must be in sorted order and not overlap.
    Zero length ranges are allowed. In this case the corresponding
    elements in the output will be zero.

    If weight is not None then we do a weighted sum. Weight
    should be a 1D array of length array.shape[0].

    If normalize is True then we divide by the length of
    each range (if weight=None) or the total weight in the
    range (if weight is not None).
    """
    
    # Ensure input is an array
    array = np.asarray(array)
    if dtype is None:
        dtype = array.dtype
    offset = np.asarray(offset).astype(np.intp)
    length = np.asarray(length).astype(np.intp)

    # Pick out non-zero length ranges
    ind = length > 0
    offset_nz = offset[ind]
    length_nz = length[ind]

    # Create mask to pick out elements to sum
    mask = np.zeros(array.shape[0], dtype=np.bool)
    assign_ranges(mask, offset_nz, length_nz, np.ones(offset_nz.shape[0], dtype=np.bool))
    
    # Allocate output array
    shape    = list(array.shape)
    shape[0] = offset.shape[0]
    result   = np.zeros(shape, dtype=dtype)
 
    if weight is None:
        # No weights
        result[ind,...] = np.add.reduceat(array[mask,...], 
                                          np.cumsum(length_nz)-length_nz,
                                          axis=0, dtype=dtype)
        if normalize:
            # Return sum divided by number of elements
            result[ind,...] = (result[ind].T / length_nz).T
    else:
        # Weighted sum.
        a_x_w     = (array[mask,...].T.astype(dtype)*weight[mask].astype(dtype)).T # transpose so we can broadcast weight
        result[ind,...] = np.add.reduceat(a_x_w, np.cumsum(length_nz)-length_nz,
                                          axis=0, dtype=dtype)
        if normalize:
            sum_w = np.add.reduceat(weight[mask], np.cumsum(length_nz)-length_nz,
                                    axis=0, dtype=dtype)
            result[ind,...] = (result[ind,...].T / sum_w).T

    return result
