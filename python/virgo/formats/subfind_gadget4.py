#!/bin/env python
#
# Classes to read binary SubFind output from Gadget-4
# (at least the version used for COCO)
#

import numpy as np
from ..util.read_binary import BinaryFile
from ..util.exceptions  import SanityCheckFailedException

class SubTabFile(BinaryFile):
    """
    Class for reading sub_tab files written by Gadget-4
    """
    def add_record(self, name, dtype, shape=()):
        """Add a Fortran record containing a single dataset to the file"""
        self.start_fortran_record()
        self.add_dataset(name, dtype, shape)
        self.end_fortran_record()

    def __init__(self, fname, id_bytes=4, *args):
        BinaryFile.__init__(self, fname, *args)

        # We need to know the data types used for particle IDs
        if id_bytes == 4:
            self.id_type = np.uint32
        elif id_bytes == 8:
            self.id_type = np.uint64
        else:
            raise ValueError("id_bytes must be 4 or 8")

        # Read file header and determine endian-ness
        self.start_fortran_record(auto_byteswap=True)
        self.add_dataset("Ngroups",       np.int32)
        self.add_dataset("Nsubgroups",    np.int32)
        self.add_dataset("Nids",          np.int32)
        self.add_dataset("TotNgroups",    np.int32)
        self.add_dataset("TotNsubgroups", np.int32)
        self.add_dataset("TotNids",       np.int32)
        self.add_dataset("NTask",         np.int32)
        self.add_dataset("padding1",      np.int32)
        self.add_dataset("Time",        np.float64)
        self.add_dataset("Redshift",    np.float64)
        self.add_dataset("HubbleParam", np.float64)
        self.add_dataset("BoxSize",     np.float64)
        self.add_dataset("Omega0",      np.float64)
        self.add_dataset("OmegaLambda", np.float64)
        self.add_dataset("flag_dp",     np.int32)
        self.add_dataset("padding2",      np.int32)
        self.end_fortran_record()

        # Check if this output uses double precision floats
        flag_dp = self["flag_dp"][...]
        if flag_dp == 0:
            self.float_type = np.float32
        else:
            self.float_type = np.float64

        # Data blocks for FoF groups
        # These are Fortran records which are only present if ngroups > 0.
        ngroups = self["Ngroups"][...]
        if ngroups > 0:
            self.add_record("GroupLen",          np.int32,        (ngroups,))
            self.add_record("GroupMass",         self.float_type, (ngroups,))
            self.add_record("GroupPos",          self.float_type, (ngroups,3))
            self.add_record("GroupVel",          self.float_type, (ngroups,3))
            self.add_record("GroupLenType",      np.int32,        (ngroups,6))
            self.add_record("GroupMassType",     self.float_type, (ngroups,6))
            self.add_record("Halo_M_Mean200",    self.float_type, (ngroups,))
            self.add_record("Halo_R_Mean200",    self.float_type, (ngroups,))
            self.add_record("Halo_M_Crit200",    self.float_type, (ngroups,))
            self.add_record("Halo_R_Crit200",    self.float_type, (ngroups,))
            self.add_record("Halo_M_TopHat200",  self.float_type, (ngroups,))
            self.add_record("Halo_R_TopHat200",  self.float_type, (ngroups,))
            self.add_record("Nsubs",             np.int32,        (ngroups,))
            self.add_record("FirstSub",          np.int32,        (ngroups,))
            
        # Data blocks for Subfind groups
        # These are Fortran records which are only present if nsubgroups > 0.
        nsubgroups = self["Nsubgroups"][...]
        if nsubgroups > 0:
            self.add_record("SubLen",             np.int32,        (nsubgroups,))
            self.add_record("SubMass",            self.float_type, (nsubgroups,))
            self.add_record("SubPos",             self.float_type, (nsubgroups,3))
            self.add_record("SubVel",             self.float_type, (nsubgroups,3))
            self.add_record("SubLenType",         np.int32,        (nsubgroups,6))
            self.add_record("SubMassType",        self.float_type, (nsubgroups,6))
            self.add_record("SubCofM",            self.float_type, (nsubgroups,3))
            self.add_record("SubSpin",            self.float_type, (nsubgroups,3))
            self.add_record("SubVelDisp",         self.float_type, (nsubgroups,))
            self.add_record("SubVmax",            self.float_type, (nsubgroups,))
            self.add_record("SubRVmax",           self.float_type, (nsubgroups,))
            self.add_record("SubHalfMassRad",     self.float_type, (nsubgroups,))
            self.add_record("SubHalfMassRadType", self.float_type, (nsubgroups,6))
            self.add_record("SubMassInRad",       self.float_type, (nsubgroups,))
            self.add_record("SubMassInRadType",   self.float_type, (nsubgroups,6))
            self.add_record("SubMostBoundID",     self.id_type,    (nsubgroups,))
            self.add_record("SubGrNr",            np.int32,        (nsubgroups,))
            self.add_record("SubParent",          np.int32,        (nsubgroups,))

