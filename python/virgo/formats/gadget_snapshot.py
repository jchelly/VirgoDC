#!/bin/env python

import numpy as np
import re
from collections import Mapping
from read_binary import BinaryFile

# Check if we have h5py
try:
    import h5py
except ImportError:
    have_hdf5 = False
else:
    have_hdf5 = True


class GadgetBinarySnapshotFile(BinaryFile):
    """
    Class which provides a h5py-like interface to a binary snapshot file
    """
    def __init__(self, fname):
        BinaryFile.__init__(self, fname)
        
        # Read the header record marker and establish endian-ness
        self.add_dataset("record_markers/header_start", np.int32)
        irec = self["record_markers/header_start"][...]
        if irec == 256:
            self.enable_byteswap(False)
        elif irec == 65536:
            self.enable_byteswap(True)
        else:
            raise IOError("Header record length is incorrect!")
            
        # Define header blocks
        self.add_attribute("Header/NumPart_ThisFile", np.int32,   (6,))
        self.add_attribute("Header/MassTable",        np.float64, (6,))
        self.add_attribute("Header/Time",             np.float64)
        self.add_attribute("Header/Redshift",         np.float64)
        self.add_attribute("Header/Flag_Sfr",         np.int32)
        self.add_attribute("Header/Flag_Feedback",    np.int32)
        self.add_attribute("Header/NumPart_Total",    np.uint32,  (6,))
        self.add_attribute("Header/Flag_Cooling",     np.int32)
        self.add_attribute("Header/NumFilesPerSnapshot", np.int32)
        self.add_attribute("Header/BoxSize",          np.float64)
        self.add_attribute("Header/Omega0",           np.float64)
        self.add_attribute("Header/OmegaLambda",      np.float64)
        self.add_attribute("Header/HubbleParam",      np.float64)
        self.add_attribute("Header/Flag_StellarAge",  np.int32)
        self.add_attribute("Header/Flag_Metals",      np.int32)
        self.add_attribute("Header/NumPart_Total_HighWord", np.uint32,  (6,))
        self.skip_bytes(256+4-self.offset)

        # Get total number of particles in this file
        npart_type = self["Header"].attrs["NumPart_ThisFile"][...]
        masstable  = self["Header"].attrs["MassTable"][...]
        npart      = sum(npart_type)

        # Check end of header marker
        self.add_dataset("record_markers/header_end", np.int32)
        if self["record_markers/header_end"][...] != 256:
            raise IOError("Header end of record marker is incorrect!")

        # Determine type of positions and add blocks
        self.add_dataset("record_markers/pos_start", np.int32)
        irec = self["record_markers/pos_start"][...]
        if npart*3*4 == irec:
            pos_type = np.float32
        elif npart*3*8 == irec:
            pos_type = np.float64
        else:
            raise IOError("Positions record length is incorrect!")
        for i in range(6):
            if npart_type[i] > 0:
                self.add_dataset("PartType%i/Coordinates" % i, pos_type, (npart_type[i],3))
        self.add_dataset("record_markers/pos_end", np.int32)
        irec = self["record_markers/pos_end"][...]
        if irec != np.dtype(pos_type).itemsize*3*npart:
            raise IOError("Positions end of record marker is incorrect!")

        # Determine type of velocities and add blocks
        self.add_dataset("record_markers/vel_start", np.int32)
        irec = self["record_markers/vel_start"][...]
        if npart*3*4 == irec:
            vel_type = np.float32
        elif npart*3*8 == irec:
            vel_type = np.float64
        else:
            raise IOError("Velocities record length is incorrect!")
        for i in range(6):
            if npart_type[i] > 0:
                self.add_dataset("PartType%i/Velocities" % i, vel_type, (npart_type[i],3))
        self.add_dataset("record_markers/vel_end", np.int32)
        irec = self["record_markers/vel_end"][...]
        if irec != np.dtype(vel_type).itemsize*3*npart:
            raise IOError("Velocities end of record marker is incorrect!")

        # Determine type of IDs and add blocks
        self.add_dataset("record_markers/ids_start", np.int32)
        irec = self["record_markers/ids_start"][...]
        if npart*4 == irec:
            ids_type = np.int32
        elif npart*8 == irec:
            ids_type = np.int64
        else:
            raise IOError("IDs record length is incorrect!")
        for i in range(6):
            if npart_type[i] > 0:
                self.add_dataset("PartType%i/ParticleIDs" % i, ids_type, (npart_type[i],))
        self.add_dataset("record_markers/ids_end", np.int32)
        irec = self["record_markers/ids_end"][...]
        if irec != np.dtype(ids_type).itemsize*npart:
            raise IOError("IDs end of record marker is incorrect!")

        # Determine type of masses and add blocks (if any)
        nmass = sum(npart_type[masstable==0.0])
        if nmass > 0:
            self.add_dataset("record_markers/mass_start", np.int32)
            irec = self["record_markers/mass_start"][...]
            if nmass*4 == irec:
                mass_type = np.float32
            elif nmass*8 == irec:
                mass_type = np.float64
            else:
                raise IOError("Mass record length is incorrect!")
            for i in range(6):
                if npart_type[i] > 0 and masstable[i] == 0:
                    self.add_dataset("PartType%i/Masses" % i, mass_type, (npart_type[i],))
            self.add_dataset("record_markers/mass_end", np.int32)
            irec = self["record_markers/mass_end"][...]
            if irec != np.dtype(mass_type).itemsize*nmass:
                raise IOError("Mass end of record marker is incorrect!")


class GadgetSnapshotFile(Mapping):
    """
    Class to read Gadget snapshot files.

    This is a thin wrapper around a GadgetBinarySnapshotFile or a h5py.File
    object, depending on the snapshot format.
    """
    def __init__(self, fname):
        snap = None
        # Try to open it as a HDF5 file
        if have_hdf5:
            try:
                snap = h5py.File(fname, "r")
                np = snap["Header"].attrs["NumPart_ThisFile"]
            except IOError:
                pass
            else:
                self.format = "HDF5"
        # If that failed, try binary type 1
        if snap is None:
            try:
                snap = GadgetBinarySnapshotFile(fname)
            except IOError:
                pass
            else:
                self.format = "BINARY1"
        if snap is not None:
            self.file  = snap
            self.fname = fname
        else:
            raise IOError("Unable to open the file as binary type 1 or HDF5!")
        
    def __len__(self):
        return self.file.__len__()

    def __getitem__(self, key):
        return self.file[key]

    def __iter__(self):
        for item in self.file:
            yield item

    def __getattr__(self, key):
        return self.file.__dict__[key]

    def __repr__(self):
        return 'GadgetSnapshotFile("%s")' % self.fname
