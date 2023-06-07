#!/bin/env python

import h5py
from virgo.formats.swift import SwiftGroup, soap_unit_registry_from_snapshot, swiftsimio_cosmology


class SOAPCatalogue(SwiftGroup):
    """
    This is a wrapper around the h5py.File object for an
    open SOAP file.

    All arguments are passed to the underlying h5py.File.
    """
    def __init__(self, *args, **kwargs):
 
        # Open the HDF5 file
        super(SOAPCatalogue, self).__init__(h5py.File(*args, **kwargs))

        # Read unit information
        self.registry = soap_unit_registry_from_snapshot(self.obj["SWIFT"])

        # Read cosmology
        self.cosmology = swiftsimio_cosmology(self.obj["SWIFT"])

