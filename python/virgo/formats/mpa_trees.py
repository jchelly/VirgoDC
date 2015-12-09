#!/bin/env python

import numpy as np
from virgo.util.read_binary import BinaryFile


class SubDescFile(BinaryFile):
    """
    Class for reading sub_desc files from B-BaseTree
    """
    def __init__(self, fname, *args):
        BinaryFile.__init__(self, fname, *args)

        self.add_dataset("TotNsubhalos", np.int32)
        TotNsubhalos = self["TotNsubhalos"][...]
        self.add_dataset("HaloIndex", np.int32, (TotNsubhalos,))
        self.add_dataset("SnapNum",   np.int32, (TotNsubhalos,))


class TreeAuxFile(BinaryFile):
    """
    Class for reading treeaux files written by L-TreeAddPosTab
    and L-TreeAddIDTab
    """
    def __init__(self, fname, id_bytes=8, *args):
        BinaryFile.__init__(self, fname, *args)

        # We need to know the data type used for particle IDs
        if id_bytes == 4:
            self.id_type = np.uint32
        elif id_bytes == 8:
            self.id_type = np.uint64
        else:
            raise ValueError("id_bytes must be 4 or 8")

        # File header
        self.add_dataset("TotHalos",  np.int32)
        self.add_dataset("Totunique", np.int32)
        self.add_dataset("Ntrees",    np.int32)
        self.add_dataset("TotSnaps",  np.int32)

        # Read header values we need
        TotHalos  = self["TotHalos"][...]
        Totunique = self["Totunique"][...]
        Ntrees    = self["Ntrees"][...]
        TotSnaps  = self["TotSnaps"][...]        
        
        # Indexing
        self.add_dataset("CountID_Snap",      np.int32, (TotSnaps,))
        self.add_dataset("OffsetID_Snap",     np.int32, (TotSnaps,))
        self.add_dataset("CountID_SnapTree",  np.int32, (TotSnaps, Ntrees))
        self.add_dataset("OffsetID_SnapTree", np.int32, (TotSnaps, Ntrees))
        self.add_dataset("Nunique",           np.int32, (TotHalos,))
        self.add_dataset("OffsetID_Halo",     np.int32, (TotHalos,))

        # Particle data arrays
        self.add_dataset("IDs", self.id_type, (Totunique,))
        self.add_dataset("Pos", np.float32,   (Totunique,3))
        self.add_dataset("Vel", np.float64,   (Totunique,3))


class TreeFile(BinaryFile):
    """
    Class for reading MPA tree files written by B-HaloTrees
    """
    def __init__(self, fname, id_bytes=8, SAVE_MASS_TAB=False, *args):
        BinaryFile.__init__(self, fname, *args)

        # We need to know the data type used for particle IDs
        if id_bytes == 4:
            self.id_type = np.uint32
        elif id_bytes == 8:
            self.id_type = np.uint64
        else:
            raise ValueError("id_bytes must be 4 or 8")

        # Define numpy record type corresponding to the halo_data struct
        fields = [
            ("Descendant",          np.int32),
            ("FirstProgenitor",     np.int32),
            ("NextProgenitor",      np.int32),
            ("FirstHaloInFOFgroup", np.int32),
            ("NextHaloInFOFgroup",  np.int32),
            ("Len",                 np.int32),
            ("M_Mean200",           np.float32),
            ("M_Crit200",           np.float32),
            ("M_TopHat",            np.float32),
            ("Pos",                 np.float32, (3,)),
            ("Vel",                 np.float32, (3,)),
            ("VelDisp",             np.float32),
            ("VMax",                np.float32),
            ("Spin",                np.float32, (3,)),
            ("MostBoundID",         self.id_type),
            ("SnapNum",             np.int32),
            ("FileNr",              np.int32),
            ("SubhaloIndex",        np.int32),
            ("SubhalohalfMass",     np.float32)
            ]
        if SAVE_MASS_TAB:
            fields.append(("SubMassTab", np.float32, (6,)))
        self.halo_data = np.dtype(fields)

        # File header
        self.add_dataset("Ntrees", np.int32)
        self.add_dataset("Nhalos", np.int32)
        Ntrees = self["Ntrees"][...]
        Nhalos = self["Nhalos"][...]

        # Number of halos per tree
        self.add_dataset("NPerTree", np.int32, (Ntrees,))

        # Array of halos
        self.add_dataset("HaloList", self.halo_data, (Nhalos,))
