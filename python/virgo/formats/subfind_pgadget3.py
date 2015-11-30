#!/bin/env python
#
# Classes to read SubFind output from P-Gadget3
# (at least the versions used for Millennium-2, Aquarius and Phoenix)
#

import numpy as np
from virgo.util.read_binary import BinaryFile
from virgo.util.exceptions  import SanityCheckFailedException
from virgo.util.read_multi import read_multi


class SubTabFile(BinaryFile):
    """
    Class for reading sub_tab files written by P-Gadget3
    """
    def __init__(self, fname, 
                 SO_VEL_DISPERSIONS=False,
                 SO_BAR_INFO=False,
                 WRITE_SUB_IN_SNAP_FORMAT=False,
                 id_bytes=8, float_bytes=4,
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
        nsubgroups = self["Nsubgroups"][...]
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
        self.add_dataset("SubGrNr",        np.int32,        (nsubgroups,))


class SubIDsFile(BinaryFile):
    """
    Class for reading sub_ids files written by P-Gadget3
    """
    def __init__(self, fname, id_bytes=8, *args):
        BinaryFile.__init__(self, fname, *args)

        # We need to know the data type used for particle IDs
        if id_bytes == 4:
            self.id_type = np.uint32
        elif id_bytes == 8:
            self.id_type = np.uint64
        
        # Define data blocks in the subhalo_tab file
        # Header
        self.add_dataset("Ngroups",    np.int32)
        self.add_dataset("TotNgroups", np.int32)
        self.add_dataset("Nids",       np.int32)
        self.add_dataset("TotNids",    np.int64)
        self.add_dataset("NTask",      np.int32)
        self.add_dataset("SendOffset", np.int32)

        # Establish endian-ness by sanity check on number of files
        nfiles = self["NTask"][...]
        if nfiles < 1 or nfiles > 65535:
            self.enable_byteswap(True)
            
        # Read header
        Nids = self["Nids"][...]

        # Add dataset with particle IDs
        self.add_dataset("GroupIDs",   self.id_type, (Nids,))


class GroupCatalogue(Mapping):
    """
    Class for reading the complete group catalogue for
    a snapshot into memory.

    This class acts as a dictionary where the keys are dataset
    names and the values are numpy arrays with the data.
    """
    def __init__(self, basedir, isnap, id_bytes=8, datasets=None):

        # Default datasets to read
        if datasets is None:
            datasets =  ["GroupLen",  "GroupOffset",  "GroupMass",  "GroupPos", 
                         "Halo_M_Mean200",  "Halo_R_Mean200",  "Halo_M_Crit200",  
                         "Halo_R_Crit200",  "Halo_M_TopHat200",  "Halo_R_TopHat200",  
                         "VelDisp_Mean200",  "VelDisp_Crit200",  "VelDisp_TopHat200",  
                         "ContaminationLen",  "ContaminationMass",  "Nsubs",  
                         "FirstSub",  "SubLen",  "SubOffset",  "SubParent",  
                         "SubMass",  "SubPos",  "SubVel",  "SubCofM",  "SubSpin",
                         "SubVelDisp",  "SubVmax",  "SubRVmax",  "SubHalfMass",  
                         "SubMostBoundID", "SubGrNr"]

        # Construct format string for file names
        fname_fmt = ("%s/groups_%03d/subhalo_tab_%03d" % (basedir, isnap, isnap)) + ".%(i)d"

        # Get number of files
        f = SubTabFile(fname_fmt % {"i":0}, id_bytes=id_bytes)
        nfiles = f["NTask"][...]
        del f
        
        # Read the catalogue data
        self._items = read_multi(SubTabFile, fname_fmt, np.arange(nfiles), datasets)

    def __getitem__(self, key):
        return self._items[key]

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        for key in self._items.keys():
            yield key
