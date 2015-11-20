#!/bin/env python
#
# Classes to read Millennium-2 output
#

import numpy as np
import subfind_pgadget3
from gadget_snapshot import GadgetBinarySnapshotFile


class SubTabFile(subfind_pgadget3.SubTabFile):
    """
    Class for reading Millennium-2 sub_tab files written by P-Gadget3
    """
    def __init__(self, fname, *args):
        subfind_pgadget3.SubTabFile.__init__(self, fname,
                                             SO_VEL_DISPERSIONS=False,
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
