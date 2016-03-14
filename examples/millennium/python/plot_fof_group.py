#!/bin/env python
#
# Plot particles in the specified FoF group
#

import matplotlib.pyplot as plt
import numpy as np

import virgo.sims.millennium as mill
from virgo.util.match import match

# These specify the group to read in
isnap = 63    # Snapshot number
ifile = 0     # Which group_tab file the group is in
ifof  = 10    # Position of group in the group_tab file, starting from zero

# Read in group lengths and offsets from the group_tab file
fname    = "/virgo/data/Millennium/snapdir_%03d/group_tab_%03d.%d" % (isnap, isnap, ifile)
grouptab = mill.GroupTabFile(fname)
fof_len    = grouptab["GroupLen"][...]
fof_offset = grouptab["GroupOffset"][...]

# Read in the IDs of particles in this group from the sub_ids file
# (note: group_ids and sub_ids files are equivalent apart
# from the ordering of IDs within each FoF group)
fname  = "/virgo/data/Millennium/postproc_%03d/sub_ids_%03d.%d" % (isnap, isnap, ifile)
subids = mill.SubIDsFile(fname)
ids    = subids["GroupIDs"][fof_offset[ifof]:fof_offset[ifof]+fof_len[ifof]] # Read just the IDs for this FoF group

# IDs in the sub_ids file consist of particle ID in the 34 least
# significant bits and hash keys in the 30 most significant bits.
# Can use bit shifting operations to separate them.
hash_key = np.right_shift(ids, 34)
ids      = np.right_shift(np.left_shift(ids,30), 30)

# hash_key contains hash key for every particle in the FoF group.
# Make array of unique hash keys to read.
keys_to_read = np.unique(hash_key)

# Now read the particle from the hash cells corresponding to these keys
snap = mill.Snapshot("/virgo/data/Millennium/", "snap_millennium", isnap)
snap_pos, snap_vel, snap_ids = snap.read_cells(keys_to_read)

# Then we need to determine which of these particles belong to
# the FoF group we're interested in. For each particle ID in the
# sub_ids file we want to find the corresponding ID in the set of
# particles we just read in.
ptr = match(ids, snap_ids) # For each entry in ids returns index of same value in snap_ids
                           # or -1 where there's no match

# Check we found all the particles
if np.any(ptr<0):
    raise Exception("Failed to find particle in specified group!")

# Check we matched up the IDs correctly
if np.any(ids != snap_ids[ptr]):
    raise Exception("Particle IDs do not match!")

# Find positions and velocities of particles in the FoF group
pos = snap_pos[ptr,:]
vel = snap_vel[ptr,:]

# Make a dotplot
plt.plot(pos[:,0], pos[:,1], "k.")
plt.gca().set_aspect("equal")
plt.xlabel("x (comoving Mpc/h)")
plt.ylabel("y (comoving Mpc/h)")
plt.title("FoF group %d from file %d of snapshot %d" % (ifof, ifile, isnap))
plt.show()

