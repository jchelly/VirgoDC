#!/bin/env python
#
# Recalculate some subhalo properties in the Millennium-2 small run
# run using the particle data.
#

import numpy as np
import matplotlib.pyplot as plt

from virgo.formats import gadget_snapshot
from virgo.formats import subfind_pgadget3
from virgo.util import read_multi, ranges, match

snapnum = 40
basedir = "/gpfs/data/jch/Millennium2/SmallRun/"

# Read particles
fname = ("%s/snapdir_%03d/snap_newtest432_%03d" % (basedir, snapnum, snapnum))+".%(i)d"
snap = read_multi.read_multi(fname, range(128), 
                             ("PartType1/Coordinates", "PartType1/Velocities", "PartType1/ParticleIDs"), 
                             file_type=gadget_snapshot.GadgetBinarySnapshotFile)
pos = snap["PartType1/Coordinates"]
vel = snap["PartType1/Velocities"]
ids = snap["PartType1/ParticleIDs"]

# Get particle mass
fname = ("%s/snapdir_%03d/snap_newtest432_%03d" % (basedir, snapnum, snapnum))+".0"
snap = gadget_snapshot.GadgetBinarySnapshotFile(fname)
mpart    = snap["Header"].attrs["MassTable"][1]
boxsize  = snap["Header"].attrs["BoxSize"]
redshift = snap["Header"].attrs["Redshift"]

# Read group information
fname = ("%s/groups_%03d/subhalo_tab_%03d" % (basedir, snapnum, snapnum))+".%(i)d"
sub_tab = read_multi.read_multi(fname, range(128), 
                                ("SubLen", "SubOffset","SubVel","SubCofM","SubSpin"), 
                                file_type=subfind_pgadget3.SubTabFile, id_bytes=4)
fname = ("%s/groups_%03d/subhalo_ids_%03d" % (basedir, snapnum, snapnum))+".%(i)d"
sub_ids = read_multi.read_multi(fname, range(128), 
                                ("GroupIDs",), 
                                file_type=subfind_pgadget3.SubIDsFile, id_bytes=4)

# Make array with subhalo index for each particle ID
nsub  = sub_tab["SubLen"].shape[0]
nids  = sub_ids["GroupIDs"].shape[0]
subnr = -np.ones(nids, dtype=np.int32)
ranges.assign_ranges(subnr, sub_tab["SubOffset"], sub_tab["SubLen"], np.arange(nsub, dtype=np.int32))

# Match IDs between subfind and snapshot
ptr = match.match(sub_ids["GroupIDs"], ids)

# Find position, velocity, ID for each particle in the subfind output
pos = pos[ptr,:]
vel = vel[ptr,:]
ids = ids[ptr]

# Sort by subhalo membership
idx = np.argsort(subnr)
pos = pos[idx,:]
vel = vel[idx,:]
ids = ids[idx]
subnr = subnr[idx]

# Discard particles in no subhalo
ind = subnr >= 0
pos = pos[ind,:]
vel = vel[ind,:]
ids = ids[ind]
subnr = subnr[ind]

# Find range of particles in each subhalo
nsub = np.amax(subnr)+1
first_particle = np.searchsorted(subnr, np.arange(nsub, dtype=np.int32), side="left")
num_particles  = np.searchsorted(subnr, np.arange(nsub, dtype=np.int32), side="right") - first_particle

# Wrap coordinates so that subhalos don't get split at the box edges
pos_rel = pos - pos[first_particle[subnr],:]            # Position relative to first particle
pos_rel = ((pos_rel + boxsize/2) % boxsize) - boxsize/2 # Do periodic wrap into -boxsize/2 to +boxsize/2
pos = pos_rel + pos[first_particle[subnr],:]            # Position may now be outside box for some particles

# Find mean position and velocity of each subhalo
a = 1.0/(1.0+redshift)
mean_pos = ranges.sum_ranges(pos, first_particle, num_particles, normalize=True, dtype=np.float64)
mean_vel = ranges.sum_ranges(vel, first_particle, num_particles, normalize=True, dtype=np.float64) * np.sqrt(a)

# Calculate angular momentum
am = ranges.sum_ranges(np.cross((pos-mean_pos[subnr,:])*a, (vel-mean_vel[subnr,:])*np.sqrt(a)), 
                       first_particle, num_particles, normalize=True, dtype=np.float64)

# Plot magnitude of angular momentum from sub_tab files and from particles
mag_am     = np.sqrt(np.sum(am**2, axis=1))
mag_am_sub = np.sqrt(np.sum(sub_tab["SubSpin"]**2, axis=1))

plt.plot(mag_am, mag_am_sub, "k.")
plt.xlabel("Angular momentum from particles")
plt.ylabel("Angular momentum from subfind output")
plt.plot((0,140), (0,140), "r-")
plt.show()
