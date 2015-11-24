#!/bin/env python
#
# Classes to read SubFind output from P-Gadget3
# (at least the versions used for Millennium-2, Aquarius and Phoenix)
#

import numpy as np
from ..util.read_binary import BinaryFile
from ..util.exceptions  import SanityCheckFailedException

class SubTabFile(BinaryFile):
    """
    Class for reading sub_tab files written by P-Gadget3
    """
    def __init__(self, fname, 
                 SO_VEL_DISPERSIONS=False,
                 SO_BAR_INFO=False,
                 WRITE_SUB_IN_SNAP_FORMAT=False,
                 id_bytes=4, float_bytes=4,
                 *args):
        BinaryFile.__init__(self, fname, *args)

        # Haven't implemented these
        if WRITE_SUB_IN_SNAP_FORMAT:
            raise NotImplementedError("Subfind outputs in type 2 binary snapshot format are not implemented")
        if SO_BAR_INFO:
            raise NotImplementedError("Subfind outputs with SO_BAR_INFO set are not implemented")

        # These parameters, which correspond to macros in Gadget's Config.sh,
        # modify the file format. The file cannot be read correctly unless these
        # are known - their values are not stored in the output.
        self.WRITE_SUB_IN_SNAP_FORMAT = WRITE_SUB_IN_SNAP_FORMAT
        self.SO_VEL_DISPERSIONS       = SO_VEL_DISPERSIONS
        self.SO_BAR_INFO              = SO_BAR_INFO

        # We also need to know the data types used for particle IDs
        # and floating point subhalo properties (again, this can't be read from the file).
        if id_bytes == 4:
            self.id_type = np.uint32
        elif id_bytes == 8:
            self.id_type = np.uint64
        else:
            raise ValueError("id_bytes must be 4 or 8")
        if float_bytes == 4:
            self.float_type = np.float32
        elif float_bytes == 8:
            self.float_type = np.float64
        else:
            raise ValueError("float_bytes must be 4 or 8")
       
        # Define data blocks in the subhalo_tab file
        # Header
        self.add_dataset("Ngroups",    np.int32)
        self.add_dataset("TotNgroups", np.int32)
        self.add_dataset("Nids",       np.int32)
        self.add_dataset("TotNids",    np.int64)
        self.add_dataset("NTask",     np.int32)
        self.add_dataset("Nsubgroups",    np.int32)
        self.add_dataset("TotNsubgroups", np.int32)

        # Establish endian-ness by sanity check on number of files
        nfiles = self["NTask"][...]
        if nfiles < 1 or nfiles > 65535:
            self.enable_byteswap(True)
            
        # FoF group information
        ngroups = self["Ngroups"][...]
        self.add_dataset("GroupLen",          np.int32,        (ngroups,))
        self.add_dataset("GroupOffset",       np.int32,        (ngroups,))
        self.add_dataset("GroupMass",         self.float_type, (ngroups,))
        self.add_dataset("GroupPos",          self.float_type, (ngroups,3))
        self.add_dataset("Halo_M_Mean200",    self.float_type, (ngroups,))
        self.add_dataset("Halo_R_Mean200",    self.float_type, (ngroups,))
        self.add_dataset("Halo_M_Crit200",    self.float_type, (ngroups,))
        self.add_dataset("Halo_R_Crit200",    self.float_type, (ngroups,))
        self.add_dataset("Halo_M_TopHat200",  self.float_type, (ngroups,))
        self.add_dataset("Halo_R_TopHat200",  self.float_type, (ngroups,))
        
        # Optional extra FoF fields
        if SO_VEL_DISPERSIONS:
            self.add_dataset("VelDisp_Mean200",    self.float_type, (ngroups,))
            self.add_dataset("VelDisp_Crit200",    self.float_type, (ngroups,))
            self.add_dataset("VelDisp_TopHat200",  self.float_type, (ngroups,))

        # FoF contamination info
        self.add_dataset("ContaminationLen",       np.int32,        (ngroups,))
        self.add_dataset("ContaminationMass",      self.float_type, (ngroups,))
        
        # Count and offset to subhalos in each FoF group
        self.add_dataset("Nsubs",                  np.int32,        (ngroups,))
        self.add_dataset("FirstSub",               np.int32,        (ngroups,))
        
        # Subhalo properties
        self.add_dataset("SubLen",     np.int32, (nsubgroups,))
        self.add_dataset("SubOffset",  np.int32, (nsubgroups,))
        self.add_dataset("SubParent",  np.int32, (nsubgroups,))
        self.add_dataset("SubMass",    self.float_type, (nsubgroups,))
        self.add_dataset("SubPos",     self.float_type, (nsubgroups,3))
        self.add_dataset("SubVel",     self.float_type, (nsubgroups,3))
        self.add_dataset("SubCofM",    self.float_type, (nsubgroups,3))
        self.add_dataset("SubSpin",    self.float_type, (nsubgroups,3))
        self.add_dataset("SubVelDisp",     self.float_type, (nsubgroups,))
        self.add_dataset("SubVmax",        self.float_type, (nsubgroups,))
        self.add_dataset("SubRVmax",       self.float_type, (nsubgroups,))
        self.add_dataset("SubHalfMass",    self.float_type, (nsubgroups,))
        self.add_dataset("SubMostBoundID", self.id_type,    (nsubgroups,))
        self.add_dataset("SubGrNr",        self.int,        (nsubgroups,))

