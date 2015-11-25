#!/bin/env python
#
# Classes to read Milli-Millennium output
#
# These are just calling the L-Gadget2 routines, specifying 4 byte IDs
# where it's not possible to determine the ID size automatically.
#

import numpy as np
import ..formats.subfind_lgadget2
from ..formats.gadget_snapshot import GadgetBinarySnapshotFile


class GroupTabFile(subfind_lgadget2.GroupTabFile):
    """
    Class for reading Millennium-1 group_tab files written by L-Gadget2
    """
    pass


class GroupIDsFile(subfind_lgadget2.GroupIDsFile):
    """
    Class for reading Millennium-1 group_tab files written by L-Gadget2
    """
    def __init__(self, fname, *args):
        subfind_lgadget2.GroupIDsFile.__init__(self, fname, id_bytes=4, *args)


class SubTabFile(subfind_lgadget2.SubTabFile):
    """
    Class for reading Millennium-1 sub_tab files written by L-SubFind
    """
    def __init__(self, fname, *args):
        subfind_lgadget2.SubTabFile.__init__(self, fname, id_bytes=4, *args)


class SubIDsFile(subfind_lgadget2.SubIDsFile):
    """
    Class for reading Millennium-1 sub_ids files written by L-SubFind
    """
    def __init__(self, fname, *args):
        subfind_lgadget2.SubIDsFile.__init__(self, fname, id_bytes=4, *args)


class SnapshotFile(subfind_lgadget2.SnapshotFile):
    """
    Class for reading Millennium-1 snapshot files
    """
    pass


class Snapshot(subfind_lgadget2.SnapshotFile):
    """
    Class for reading parts of a Millennium snapshot
    using the hash table.
    """
    pass


