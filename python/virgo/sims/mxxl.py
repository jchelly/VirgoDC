#!/bin/env python
#
# Classes to read Millennium-XXL output
# These call the lgadget3 reading code with suitable default parameters.
#
# There are also classes for reading subhalo descendant information and
# density fields which are specific to MXXL.
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


class FieldFile(BinaryFile):
    def __init__(self, fname, *args):
        BinaryFile.__init__(self, fname, *args)

        # Read header and check if we need byte swapping
        irec = self.read_and_skip(np.int32)
        self.enable_byteswap(irec != 8)
        self.add_dataset("NTask",   np.int32)
        self.add_dataset("BoxSize", np.float64)
        self.add_dataset("nn",      np.int32)
        self.add_dataset("isw_slabs_per_task", np.int32)
        irec = self.read_and_skip(np.int32)

        # Set up the grid dataset
        nn = self["nn"][()]
        isw_slabs_per_task = self["isw_slabs_per_task"][()]
        irec = self.read_and_skip(np.int32)
        if irec != 4*isw_slabs_per_task*nn*nn:
            raise IOError("Start of grid record has wrong length in density field file")
        self.add_dataset("grid", np.float32, (isw_slabs_per_task,nn,nn))
        irec = self.read_and_skip(np.int32)
        if irec != 4*isw_slabs_per_task*nn*nn:
            raise IOError("End of grid record has wrong length in density field file")
        
