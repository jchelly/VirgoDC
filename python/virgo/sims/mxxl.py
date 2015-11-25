#!/bin/env python
#
# Classes to read Millennium-XXL output
#

import numpy as np
import ..formats.subfind_lgadget3
from ..formats.gadget_snapshot  import GadgetBinarySnapshotFile
from ..util import BinaryFile

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
    def __init__(self, fname, *args):
        subfind_lgadget3.SubTabFile.__init__(self, fname,
                                             SO_VEL_DISPERSIONS=True,
                                             SO_BAR_INFO=False,
                                             WRITE_SUB_IN_SNAP_FORMAT=False,
                                             id_bytes=8, float_bytes=4,
                                             *args)


class SnapshotFile(GadgetBinarySnapshotFile):
    """
    Class for reading Millennium-2 snapshot files.
    These are really just Gadget binary snapshots.
    """
    pass
