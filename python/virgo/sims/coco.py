#!/bin/env python
#
# Classes to read COCO output
#
# COCO has Gadget type 1 binary snapshots and Gadget-4 subfind
# output files.
#

import numpy as np
import virgo.formats.subfind_gadget4 as subfind_gadget4
from virgo.formats.gadget_snapshot import GadgetBinarySnapshotFile

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
    Class for extracting groups and subgroups from COCO 
    output with group ordered snapshot files.
    """
    def __init__(self, basedir, basename, isnap):
        subfind_gadget4.GroupOrderedSnapshot.__init__(self, 
                                                      basedir, 
                                                      basename, 
                                                      isnap, 
                                                      id_bytes=8,
                                                      *args)


class GroupCatalogue(subfind_gadget4.GroupCatalogue):
    """
    Class for reading the complete group catalogue for
    a COCO snapshot into memory.

    This class acts as a dictionary where the keys are dataset
    names and the values are numpy arrays with the data.
    """
    def __init__(self, basedir, isnap, datasets=None):
        subfind_gadget4.GroupOrderedSnapshot.__init__(self, 
                                                      basedir, 
                                                      basename, 
                                                      isnap, 
                                                      id_bytes=8,
                                                      *args)
