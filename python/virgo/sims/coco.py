#!/bin/env python
#
# Classes to read COCO output
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
