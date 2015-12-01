#!/bin/env python
#
# Classes to read Millennium-2 output
#
# These call the pgadget3 reading code with appropriate default
# parameters.
#

import numpy as np
import virgo.formats.subfind_pgadget3 as subfind_pgadget3
from virgo.formats.gadget_snapshot import GadgetBinarySnapshotFile


class SubTabFile(subfind_pgadget3.SubTabFile):
    """
    Class for reading Millennium-2 sub_tab files written by P-Gadget3
    """
    def __init__(self, fname, 
                 SO_VEL_DISPERSIONS=False,
                 SO_BAR_INFO=False,
                 WRITE_SUB_IN_SNAP_FORMAT=False,
                 id_bytes=8, float_bytes=4):
        subfind_pgadget3.SubTabFile.__init__(self, fname,
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


class GroupCatalogue(subfind_pgadget3.GroupCatalogue):
    """
    Class for reading the complete group catalogue for
    a Millennium-2 snapshot into memory.

    This class acts as a dictionary where the keys are dataset
    names and the values are numpy arrays with the data.
    """
    def __init__(self, basedir, isnap, datasets=None,
                 SO_VEL_DISPERSIONS=False,
                 SO_BAR_INFO=False,
                 WRITE_SUB_IN_SNAP_FORMAT=False,
                 id_bytes=8, float_bytes=4):
        subfind_pgadget3.GroupCatalogue.__init__(self, 
                                                 basedir, 
                                                 isnap,
                                                 datasets=datasets,
                                                 SO_VEL_DISPERSIONS=SO_VEL_DISPERSIONS,
                                                 SO_BAR_INFO=SO_BAR_INFO,
                                                 WRITE_SUB_IN_SNAP_FORMAT=WRITE_SUB_IN_SNAP_FORMAT,
                                                 id_bytes=id_bytes, float_bytes=float_bytes)


class GroupOrderedSnapshot(subfind_pgadget3.GroupOrderedSnapshot):
    """
    Class for extracting groups and subgroups from Millennium-2 
    output with group ordered snapshot files.
    """
    def __init__(self, basedir, basename, isnap,
                 SO_VEL_DISPERSIONS=False,
                 SO_BAR_INFO=False,
                 WRITE_SUB_IN_SNAP_FORMAT=False,
                 id_bytes=8, float_bytes=4):
        subfind_pgadget3.GroupOrderedSnapshot.__init__(self, 
                                                       basedir, 
                                                       basename, 
                                                       isnap,
                                                       SO_VEL_DISPERSIONS=SO_VEL_DISPERSIONS,
                                                       SO_BAR_INFO=SO_BAR_INFO,
                                                       WRITE_SUB_IN_SNAP_FORMAT=WRITE_SUB_IN_SNAP_FORMAT,
                                                       id_bytes=id_bytes, float_bytes=float_bytes)

