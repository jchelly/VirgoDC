#!/bin/env python
#
# Read a snapshot from the Millennium-WMAP7 run
# using h5py
#

import numpy as np
import h5py

# Name of the file to read
fname = "/gpfs/data/Millgas/data/dm/500/snapdir_061/500_dm_061.0.hdf5"

# Open file
f = h5py.File(fname, "r")

# Read header
# Can get full list of attributes with f["Header"].keys()
boxsize = f["Header"].attrs["BoxSize"]
print "Box size is ", boxsize, " comoving Mpc/h"

numpart_thisfile = f["Header"].attrs["NumPart_ThisFile"]
print "Number of particles in this file is ", numpart_thisfile[1]

# Read positions, velocities and IDs
pos = f["PartType1/Coordinates"][...]
vel = f["PartType1/Velocities"][...]
ids = f["PartType1/ParticleIDs"][...]

# Close HDF5 file
f.close()
