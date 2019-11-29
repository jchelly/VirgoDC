#!/bin/env python
#
# Classes to read Millennium-XXL output
# These call the lgadget3 reading code with suitable default parameters.
#

import numpy as np
import virgo.formats.subfind_lgadget3 as subfind_lgadget3
from virgo.formats.gadget_snapshot  import GadgetBinarySnapshotFile
from virgo.util.read_binary import BinaryFile


class SubDescFile(BinaryFile):
    """
    Class for reading MXXL subhalo descendant information
    """
    def __init__(self, fname, *args):
        BinaryFile.__init__(self, fname, *args)
        self.add_dataset("Nsubgroups", np.int32)
        nsub = self["Nsubgroups"][...]
        self.add_dataset("SubMostBoundID", np.uint64, (nsub,))
        self.add_dataset("Desc_FileNr",    np.int32,  (nsub,))
        self.add_dataset("Desc_SubIndex",  np.int32,  (nsub,))


class SubTabFile(subfind_lgadget3.SubTabFile):
    """
    Class for reading Millennium-XXL sub_tab files written by L-Gadget3
    """
    def __init__(self, fname, 
                 SO_VEL_DISPERSIONS=True,
                 SO_BAR_INFO=False,
                 WRITE_SUB_IN_SNAP_FORMAT=False,
                 id_bytes=8, float_bytes=4):
        subfind_lgadget3.SubTabFile.__init__(self, fname,
                                             SO_VEL_DISPERSIONS=SO_VEL_DISPERSIONS,
                                             SO_BAR_INFO=SO_BAR_INFO,
                                             WRITE_SUB_IN_SNAP_FORMAT=WRITE_SUB_IN_SNAP_FORMAT,
                                             id_bytes=id_bytes, float_bytes=float_bytes)


class SnapshotFile(GadgetBinarySnapshotFile):
    """
    Class for reading Millennium-2 snapshot files.
    These are really just Gadget binary snapshots.
    """
    pass
