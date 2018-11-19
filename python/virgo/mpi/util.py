#!/bin/env python


def broadcast_dtype_and_dims(arr, comm):
    """
    Determine dtype of the specified array, arr, which may be 
    None on ranks which have no local elements.

    Also returns shape[1:] for the array.

    Will return (None, None) if array is None on all tasks.
    """
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


def replace_none_with_zero_size(arr, comm):
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
    
    arr_dtype, arr_shape = broadcast_dtype_and_dims(arr, comm)

    if arr_dtype is None:
        return None
    elif arr is None:
        return np.ndarray([0,]+list(arr_shape), dtype=arr_dtype)
    else:
        return arr
