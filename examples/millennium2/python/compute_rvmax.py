#!/bin/env python
#
# Compute vmax, rvmax for a subfind group in Millennium-2
# using the subfind group ordered snapshots.
#
import numpy as np
import matplotlib.pyplot as plt

from virgo.formats import gadget_snapshot
from virgo.formats import subfind_pgadget3
from virgo.util import read_multi, ranges, match

G_SI         = 6.67408e-11
MPC_IN_M     = 3.086e22
MSOLAR_IN_KG = 1.98855e30

basedir  = "/virgo/data/Millennium2/BigRun/"
basename = "snap_newMillen_subidorder"
snapnum  = 40
isub = 43

# Read group catalogue
cat = subfind_pgadget3.GroupCatalogue(basedir, snapnum)

# Determine FoF group index which contains subhalo isub
ifof = cat["SubGrNr"][isub]

# Read particle data for FoF group ifof
snap = subfind_pgadget3.GroupOrderedSnapshot(basedir, basename, snapnum)
data = snap.read_fof_group(ifof)

# Get box size and particle mass from one of the snapshot files
snapfile = snap.open_snap_file(0)
boxsize  = snapfile["Header"].attrs["BoxSize"]
mpart    = snapfile["Header"].attrs["MassTable"][1]
redshift = snapfile["Header"].attrs["Redshift"]
a        = 1.0/(1.0+redshift)

pos   = data["Coordinates"]
vel   = data["Velocities"]
ids   = data["ParticleIDs"]
subnr = data["SubNr"]

# Get positions relative to subhalo centre 
pos_rel = pos - cat["SubPos"][isub,:]
pos_rel = ((pos_rel+boxsize/2) % boxsize) - boxsize/2

# Discard particles not in subhalo
ind = (subnr == isub)
pos_rel = pos_rel[ind,:]

# Get sorted radii of particles
radius = np.sqrt(np.sum(pos_rel**2, axis=1))
idx    = np.argsort(radius)
radius = radius[idx]

# Calculate circular velocity
np_interior    = np.arange(radius.shape[0], dtype=np.int32) # Number of particles interior to each particle
m_interior_si  = np_interior.astype(np.float64) * mpart * 1.0e10 * MSOLAR_IN_KG
radius_si      = radius.astype(np.float64)*a*MPC_IN_M
ind            = radius > 0.0
vcirc          = np.zeros(np_interior.shape, dtype=np.float64)
vcirc[ind]     = np.sqrt(G_SI*m_interior_si[ind]/radius_si[ind]) / 1000.0 # Convert to km/sc

imax = np.argmax(vcirc)
print "RVmax (from particles) = ", radius[imax]*a # radii from snapshot are comoving
print "RVmax (from catalogue) = ", cat["SubRVmax"][isub]

print "Vmax (from particles) = ", vcirc[imax]
print "Vmax (from catalogue) = ", cat["SubVmax"][isub]

