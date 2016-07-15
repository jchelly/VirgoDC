#!/bin/env python

import numpy as np
import re
from collections import Mapping
from  virgo.util.read_binary import BinaryFile

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

    By default this can read the following quantities:

      Coordinates
      Velocities
      Masses
      ParticleIDs
      InternalEnergy
      Density
      SmoothingLength

    Extra datasets can be specified via the "extra" parameter,
    which should be a sequence of tuples. Each tuple consists of

    (name, typestr, shape, ptypes)
    
    where the components are

    name    : name of the dataset
    typestr : string, either "float" or "int" depending on type of data.
              Number of bytes per quantity is determined from record markers.
    shape   : shape of the data for ONE particle:
              should be () for scalar quanities, (3,) for vector quantities
    ptypes  : sequence of six booleans, true for particle types which have this quantity

    These datasets should be specified in the order in which they
    appear in the file.

    """
    def __init__(self, fname, extra=None):
        BinaryFile.__init__(self, fname)

        # Read the header record marker and establish endian-ness
        irec = self.read_and_skip(np.int32)
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
        irec = self.read_and_skip(np.int32)
        if irec != 256:
            raise IOError("Header end of record marker is incorrect!")

        # Make full list of datasets to read
        all_datasets = (
            ("Coordinates",     "float", (3,), (True,)*6),
            ("Velocities",      "float", (3,), (True,)*6),
            ("ParticleIDs",     "int",   (),   (True,)*6),
            ("Masses",          "float", (),   masstable==0),
            ("InternalEnergy",  "float", (),   (True, False, False, False, False, False)),
            ("Density",         "float", (),   (True, False, False, False, False, False)),
            ("SmoothingLength", "float", (),   (True, False, False, False, False, False)),
            )
        # Add any user specified fields
        if extra is not None:
            all_datasets += extra
        
        # Determine what datasets are present in this file
        for(name, typestr, shape, ptypes) in all_datasets:

            # Calculate number of particles we expect in this dataset
            nextra = sum(npart_type[np.asarray(ptypes, dtype=np.bool)])

            # Calculate number of numbers per particle
            n_per_part = 1
            for s in shape:
                n_per_part *= s

            # Read start of record marker
            irec = self.read_and_skip(np.int32)

            # Determine bytes per quantitiy
            nbytes = irec / (n_per_part*nextra)
            if (nbytes != 4 and nbytes != 8) or nbytes*n_per_part*nextra != irec:
                raise IOError("%s record has unexpected length!" % name)

            # Determine data type for this record
            if typestr == "int":
                if nbytes==4:
                    dtype = np.int32
                else:
                    dtype = np.int64
            elif typestr == "float":
                if nbytes==4:
                    dtype = np.float32
                else:
                    dtype = np.float64
            else:
                raise ValueError("typestr parameter should be 'int' or 'float'")

            # Loop over particle types and add datasets
            for i in range(6):
                if ptypes[i] and npart_type[i] > 0:
                    full_shape = (npart_type[i],)+tuple(shape)
                    self.add_dataset("PartType%i/%s" % (i, name), dtype, full_shape)

            # Read end of record marker
            irec = self.read_and_skip(np.int32)
            if irec != n_per_part * np.dtype(dtype).itemsize * nextra:
                raise IOError("%s end of record marker is incorrect!" % name)

                        
def open(fname, extra=None):
    """
    Open a Gadget snapshot file which may be in binary or HDF5 format.

    Parameter "extra" specifies any additional datasets
    to read in the case of a binary snapshot.
    """
    snap = None
    # Try to open it as a HDF5 file
    if have_hdf5:
        try:
            snap = h5py.File(fname, "r")
            np = snap["Header"].attrs["NumPart_ThisFile"]
        except IOError:
            pass
        else:
            return snap
    # If that failed, try binary type 1
    if snap is None:
        try:
            snap = GadgetBinarySnapshotFile(fname, extra=extra)
        except IOError:
            pass
        else:
            return snap
    raise IOError("Unable to open the file %s as binary type 1 or HDF5!" % fname)
