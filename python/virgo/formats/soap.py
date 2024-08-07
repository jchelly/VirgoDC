#!/bin/env python

import h5py
from virgo.formats.swift import SwiftFile


class SOAPCatalogue(SwiftFile):
    """
    This is a wrapper around the h5py.File object for an
    open SOAP file.
    """
    def __init__(self, *args, mode="soap", **kwargs): 
        metadata_path="/"
        super(SOAPCatalogue, self).__init__(mode, metadata_path, *args, **kwargs)

