#!/bin/env python
#
# Classes to read SubFind output from L-Gadget3
# (e.g. MXXL, P-Millennium)
#

import numpy as np
from virgo.util.read_binary import BinaryFile
from virgo.util.exceptions  import SanityCheckFailedException

class SubTabFile(BinaryFile):
    """
    Class for reading sub_tab files written by L-Gadget3
    """
    def __init__(self, fname, 
                 SO_VEL_DISPERSIONS=False,
                 SUB_SHAPES=False,
                 id_bytes=4, float_bytes=4,
                 *args):
        BinaryFile.__init__(self, fname, *args)

        # These parameters, which correspond to macros in Gadget's Config.sh,
        # modify the file format. The file cannot be read correctly unless these
        # are known - their values are not stored in the output.
        self.SO_VEL_DISPERSIONS = SO_VEL_DISPERSIONS
        self.SUB_SHAPES         = SUB_SHAPES

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
        self.add_dataset("TotNgroups", np.int64)
        self.add_dataset("Nids",       np.int32)
        self.add_dataset("TotNids",    np.int64)
        self.add_dataset("NTask",     np.int32)
        self.add_dataset("Nsubgroups",    np.int32)
        self.add_dataset("TotNsubgroups", np.int64)

        # Establish endian-ness by sanity check on number of files
        nfiles = self["NTask"][...]
        if nfiles < 1 or nfiles > 65535:
            self.enable_byteswap(True)
            
        # FoF group information
        ngroups = self["Ngroups"][...]
        self.add_dataset("GroupLen",     np.int32,        (ngroups,))
        self.add_dataset("GroupOffset",  np.int32,        (ngroups,))
        self.add_dataset("GroupNr",      np.int64,        (ngroups,))
        self.add_dataset("GroupCofM",    self.float_type, (ngroups,3))
        self.add_dataset("GroupVel",     self.float_type, (ngroups,3))
        self.add_dataset("GroupPos",     self.float_type, (ngroups,3))
        self.add_dataset("Halo_M_Mean200",    self.float_type, (ngroups,))
        self.add_dataset("Halo_M_Crit200",    self.float_type, (ngroups,))
        self.add_dataset("Halo_M_TopHat200",  self.float_type, (ngroups,))
        self.add_dataset("GroupVelDisp",      self.float_type, (ngroups,))
        
        # Optional extra FoF fields
        if SO_VEL_DISPERSIONS:
            self.add_dataset("VelDisp_Mean200",    self.float_type, (ngroups,))
            self.add_dataset("VelDisp_Crit200",    self.float_type, (ngroups,))
            self.add_dataset("VelDisp_TopHat200",  self.float_type, (ngroups,))
        
        # Count and offset to subhalos in each FoF group
        self.add_dataset("Nsubs",                  np.int32,        (ngroups,))
        self.add_dataset("FirstSub",               np.int32,        (ngroups,))
        
        # Subhalo properties
        nsubgroups = self["Nsubgroups"][...]
        self.add_dataset("SubLen",     np.int32, (nsubgroups,))
        self.add_dataset("SubOffset",  np.int32, (nsubgroups,))
        self.add_dataset("SubGrNr",    np.int64, (nsubgroups,))
        self.add_dataset("SubNr",      np.int64, (nsubgroups,))
        self.add_dataset("SubPos",     self.float_type, (nsubgroups,3))
        self.add_dataset("SubVel",     self.float_type, (nsubgroups,3))
        self.add_dataset("SubCofM",    self.float_type, (nsubgroups,3))
        self.add_dataset("SubSpin",    self.float_type, (nsubgroups,3))
        self.add_dataset("SubVelDisp",     self.float_type, (nsubgroups,))
        self.add_dataset("SubVmax",        self.float_type, (nsubgroups,))
        self.add_dataset("SubRVmax",       self.float_type, (nsubgroups,))
        self.add_dataset("SubHalfMass",    self.float_type, (nsubgroups,))
        if SUB_SHAPES:
            self.add_dataset("SubShape", self.float_type, (nsubgroups,6))
        self.add_dataset("SubBindingEnergy",   self.float_type, (nsubgroups,))
        self.add_dataset("SubPotentialEnergy", self.float_type, (nsubgroups,))
        self.add_dataset("SubProfile",         self.float_type, (nsubgroups,9))
        
    def sanity_check(self):

        # Check file has the expected size (e.g. in case ID type is wrong)
        if not(self.all_bytes_used()):
            raise SanityCheckFailedException("File size is incorrect!")
        
        totnids       = self["TotNids"][...]
        totngroups    = self["TotNgroups"][...]
        totnsubgroups = self["TotNsubgroups"][...]
        
        # Checks on header
        if totnids < 0 or totngroups < 0 or totnsubgroups < 0:
            raise SanityCheckFailedException("Negative number of groups/subgroups/IDs")

        # Checks on group properties
        if np.any(self["GroupLen"][...] < 0):
            raise SanityCheckFailedException("Found group with non-positive length!")
        if totnids < 2**31:
            # Offsets overflow if we have too many particles (e.g. Millennium-2), 
            # so this test is expected to fail in that case
            if np.any(self["GroupOffset"][...] < 0) or np.any(self["GroupOffset"][...]+self["GroupLen"][...] > totnids):
                raise SanityCheckFailedException("Found group with offset out of range!")

        # Check pointers from groups to subgroups are sane
        firstsub = self["FirstSub"][...]
        nsubs    = self["Nsubs"][...]
        if np.any(nsubs < 0):
            raise SanityCheckFailedException("Negative number of subgroups in group!")
        ind = nsubs > 0
        if np.any(firstsub[ind] < 0) or np.any(firstsub[ind]+nsubs[ind] > totnsubgroups):
            raise SanityCheckFailedException("Found group with subgroup index out of range!")

        # Check some group properties that we expect to be finite and non-negative
        for prop in ("Halo_M_Mean200", "Halo_M_Crit200", "Halo_M_TopHat200", "GroupVelDisp"):
            data = self[prop][...]
            if not(np.all(np.isfinite(data))):
                raise Exception("Found non-finite value in dataset %s" % prop)
            if not(np.all(data >= 0.0)):
                raise Exception("Found negative value in dataset %s" % prop)

        # Checks on subgroup properties
        if np.any(self["SubLen"][...] < 0):
            raise SanityCheckFailedException("Found subgroup with non-positive length!")
        if totnids < 2**31:
            # Offsets overflow if we have too many particles (e.g. Millennium-2),
            # so this test is expected to fail in that case
            if np.any(self["SubOffset"][...] < 0) or np.any(self["SubOffset"][...]+self["SubLen"][...] > totnids):
                raise SanityCheckFailedException("Found group with offset out of range!")

        # Check some subgroup properties that we expect to be finite and (in some cases) non-negative
        for prop in ("SubPos","SubVel","SubCofM","SubSpin","SubVelDisp",
                     "SubVmax","SubRVmax","SubHalfMass"):
            data = self[prop][...]
            if not(np.all(np.isfinite(data))):
                raise Exception("Found non-finite value in dataset %s" % prop)
            if not(np.all(data >= 0.0)) and prop in ("SubMass","SubVelDisp","SubVmax","SubRVmax","SubHalfMass"):
                raise Exception("Found negative value in dataset %s" % prop)

        # Check SubGrNr is in range and in the expected order
        subgrnr = self["SubGrNr"][...]
        if np.any(subgrnr<0) or np.any(subgrnr>=totngroups):
            raise SanityCheckFailedException("Subgroup's SubGrNr out of range!")
        if subgrnr.shape[0] > 1:
            if np.any(subgrnr[1:] < subgrnr[:-1]):
                raise SanityCheckFailedException("Subgroup SubGrNr's are not in ascending order!")






class SubIDsFile(BinaryFile):
    """
    Class for reading sub_ids files written by L-Gadget3
    (e.g. P-Millennium outputs)
    """
    def __init__(self, fname, 
                 id_bytes=4, *args):
        BinaryFile.__init__(self, fname, *args)

        # We need to know the data type used for particle IDs
        if id_bytes == 4:
            self.id_type = np.uint32
        elif id_bytes == 8:
            self.id_type = np.uint64
 
        # Define data blocks in the subhalo_tab file
        # Header
        self.add_dataset("Ngroups",    np.int32)
        self.add_dataset("TotNgroups", np.int64)
        self.add_dataset("Nids",       np.int32)
        self.add_dataset("TotNids",    np.int64)
        self.add_dataset("NTask",     np.int32)
        self.add_dataset("NidsPrevious", np.int64)

        # Establish endian-ness by sanity check on number of files
        nfiles = self["NTask"][...]
        if nfiles < 1 or nfiles > 65535:
            self.enable_byteswap(True)
            
        # Read header
        Nids = self["Nids"][...]

        # Add dataset with particle IDs
        self.add_dataset("GroupIDs",   self.id_type, (Nids,))

    def sanity_check(self):

        # Check file has the expected size (e.g. in case ID type is wrong)
        if not(self.all_bytes_used()):
            raise SanityCheckFailedException("File size is incorrect!")

        # Check that IDs don't contain duplicates
        ids = self["GroupIDs"][...]
        idx, counts = np.unique(ids, return_counts=True)
        if np.any(counts != 1):
            raise SanityCheckFailedException("Found duplicate IDs!")
