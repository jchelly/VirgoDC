#!/bin/env python

import numpy as np
from collections import OrderedDict

def read_multi(file_type, format_str, idx, datasets, axis=0, *args, **kwargs):
    """
    Read and concatenate arrays from the specified set of files.

    file_type : class to use to read each file
    format_str: format string such that file names are given by
                format_str % {'i':i}
    idx       : sequence with indexes of files to read
    datasets  : sequence with names of datasets to read
    axis      : axis along which to concatenate arrays

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

        # Open file, passing in any extra args
        f = file_type(format_str % {'i':i}, *args, **kwargs)

        # Read the datasets
        for name in datasets:
            if name in f:
                result[name].append(f[name][...])

    # Concatenate arrays
    for name in datasets:
        result[name] = np.concatenate(result[name], axis=axis)

    return result
