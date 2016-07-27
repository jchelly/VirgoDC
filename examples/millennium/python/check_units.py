#!/bin/env python
#
# Recalculate some subhalo properties in the Milli-Millennium
# run using the particle data.
#

from virgo.formats import gadget_snapshot
from virgo.formats import subfind_lgadget2
from virgo.util import read_multi, ranges, match

snapnum = 40
basedir = "/gpfs/data/jch/MilliMillennium/Snapshots/"

# Read particles
fname = ("%s/snapdir_%03d/snap_milli_%03d" % (basedir, snapnum, snapnum))+".%(i)d"
snap = read_multi.read_multi(fname, range(8), 
                             ("PartType1/Coordinates", "PartType1/Velocities", "PartType1/ParticleIDs"), 
                             file_type=gadget_snapshot.GadgetBinarySnapshotFile)
pos = snap["PartType1/Coordinates"]
vel = snap["PartType1/Velocities"]
ids = snap["PartType1/ParticleIDs"]

# Get particle mass
fname = ("%s/snapdir_%03d/snap_milli_%03d" % (basedir, snapnum, snapnum))+".0"
snap = gadget_snapshot.GadgetBinarySnapshotFile(fname)
mpart    = snap["Header"].attrs["MassTable"][1]
boxsize  = snap["Header"].attrs["BoxSize"]
redshift = snap["Header"].attrs["Redshift"]

# Read group information
groupids  = []
subnr = []
offset = 0
sub_vel = []
sub_spin = []
for i in range(8):

    # Read subhalo offsets and lengths and particle IDs
    fname = ("%s/postproc_%03d/sub_tab_%03d" % (basedir, snapnum, snapnum))+".%(i)d"
    sub_tab = subfind_lgadget2.SubTabFile(fname % {"i" : i})
    sublen = sub_tab["SubLen"][...]
    suboff = sub_tab["SubOffset"][...]
    sub_vel.append(sub_tab["SubVel"][...])
    sub_spin.append(sub_tab["SubSpin"][...])
    fname = ("%s/postproc_%03d/sub_ids_%03d" % (basedir, snapnum, snapnum))+".%(i)d"
    sub_ids = subfind_lgadget2.SubIDsFile(fname % {"i" : i})
    groupids.append(sub_ids["GroupIDs"][...])

    # Make array with subhalo index for each particle ID
    nsub = sublen.shape[0]
    nids = groupids[-1].shape[0]
    nr = -np.ones(nids, dtype=np.int32)
    ranges.assign_ranges(nr, suboff, sublen, np.arange(nsub, dtype=np.int32)+offset)
    subnr.append(nr)

    offset += nsub

groupids = np.concatenate(groupids)
subnr    = np.concatenate(subnr)
sub_vel  = np.concatenate(sub_vel, axis=0)
sub_spin = np.concatenate(sub_spin, axis=0)

# Discard hash keys from IDs
groupids = (groupids << 34) >> 34

# Match IDs between subfind and snapshot
ptr = match.match(groupids, ids)

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
am = ranges.sum_ranges(np.cross((pos-mean_pos[subnr,:])*a, (vel-mean_vel[subnr,:])*sqrt(a)), 
                       first_particle, num_particles, normalize=True, dtype=np.float64)

