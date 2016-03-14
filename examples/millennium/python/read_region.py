#!/bin/env python
#
# Read particle positions, velocities and IDs
# for a region in a Millennium simulation snapshot
#

import virgo.sims.millennium as mill

# Snapshot number to read
isnap = 63

# Region to read: xmin, xmax, ymin, ymax, zmin, zmax
coords = (10.0, 20.0, 
          10.0, 20.0, 
          10.0, 20.0)

# Open the snapshot
snap = mill.Snapshot("/virgo/data/Millennium/", "snap_millennium", isnap)

# Read the particle data for this region
pos, vel, ids = snap.read_region(coords)

