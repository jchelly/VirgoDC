#!/bin/env python
#
# Classes to read COCO output
#

import numpy as np
import virgo.formats.subfind_gadget4 as subfind_gadget4
from virgo.formats.gadget_snapshot import GadgetBinarySnapshotFile
from virgo.util.read_multi import read_multi
from collections import Mapping

class SubTabFile(subfind_gadget4.SubTabFile):
    """
    Class for reading COCO sub_tab files written by Gadget-4
    """
    def __init__(self, fname, *args):
        subfind_gadget4.SubTabFile.__init__(self, fname, id_bytes=8,
                                            *args)

class SnapshotFile(GadgetBinarySnapshotFile):
    """
    Class for reading COCO snapshot files.
    These are really just Gadget binary snapshots.
    """
    pass


class GroupOrderedSnapshot(subfind_gadget4.GroupOrderedSnapshot):
    """
    Class for extracting groups and subgroups from Gadget-4 
    output with group ordered snapshot files.
    """
    pass


class GroupCatalogue(Mapping):
    """
    Class for reading the complete group catalogue for
    a snapshot into memory.

    This class acts as a dictionary where the keys are dataset
    names and the values are numpy arrays with the data.
    """
    def __init__(self, basedir, isnap):

        # Datasets to read
        datasets = ["GroupLen", "GroupMass", "GroupPos", "GroupVel", "GroupLenType", "GroupMassType", 
                    "Halo_M_Mean200",   "Halo_R_Mean200", 
                    "Halo_M_Crit200",   "Halo_R_Crit200", 
                    "Halo_M_TopHat200", "Halo_R_TopHat200", 
                    "Nsubs", "FirstSub", "SubLen", "SubMass", "SubPos", "SubVel", 
                    "SubLenType", "SubMassType", "SubCofM", "SubSpin", "SubVelDisp", 
                    "SubVmax", "SubRVmax", "SubHalfMassRad", "SubHalfMassRadType", 
                    "SubMassInRad", "SubMassInRadType", "SubMostBoundID", "SubGrNr", "SubParent"]

        # Construct format string for file names
        fname_fmt = ("%s/groups_%03d/fof_subhalo_tab_%03d" % (basedir, isnap, isnap)) + ".%(i)d"

        # Get number of files
        f = SubTabFile(fname_fmt % {"i":0})
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
