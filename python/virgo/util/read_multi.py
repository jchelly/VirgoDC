#!/bin/env python

try:
    import h5py
except ImportError:
    h5py=None

import numpy as np
from collections import OrderedDict

def read_multi(format_str, idx, datasets, file_type=None, axis=0, *args, **kwargs):
    """
    Read and concatenate arrays from the specified set of files.

    file_type : class to use to read each file. Assume hdf5 file if file_type=None.
    format_str: format string such that file names are given by
                format_str % {'i':i}
    idx       : sequence with indexes of files to read
    datasets  : sequence with names of datasets to read
    axis      : axis along which to concatenate arrays

    args and kwargs are used to pass all extra arguments to the file_type
    object, in case extra parameters are needed to open the file.

    It is assumed that the class file_type provides a h5py like interface
    to the data, so a h5py.File or a read_binary.BinaryFile may be used
    here.
    """

    if args is None:
        args = []
    
    # Create dictionary of lists to store output
    result = OrderedDict()
    for name in datasets:
        result[name] = []

    # Loop over files
    for i in idx:

        # Open file
        if file_type is None:
            # Type not specified, assume hdf5
            f = h5py.File(format_str % {'i':i}, "r")
        else:
            # Have been given a class for reading this file type
            f = file_type(format_str % {'i':i}, *args, **kwargs)

        # Read the datasets
        for name in datasets:
            if name in f:
                result[name].append(f[name][...])

    # Concatenate arrays
    for name in datasets:
        result[name] = np.concatenate(result[name], axis=axis)

    return result
