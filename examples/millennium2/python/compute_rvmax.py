#!/bin/env python
#
# Compute vmax, rvmax for a subfind group in Millennium-2
#
import numpy as np
import matplotlib.pyplot as plt

from virgo.formats import gadget_snapshot
from virgo.formats import subfind_pgadget3
from virgo.util import read_multi, ranges, match

snapnum  = 67
basedir  = "/data/millennium2/millennium-2/"
basename = "snap_newMillen_subidorder"

# Read group catalogue
cat = subfind_pgadget3.GroupCatalogue(basedir, snapnum)

# Read particle data for FoF group 0
snap = subfind_pgadget3.GroupOrderedSnapshot(basedir, basename, snapnum)
data = snap.read_fof_group(0)

# Extract particles in first subfind group
ind = data["SubNr"]==0
pos = data["Coordinates"][ind,:]
vel = data["Velocities"][ind,:]
ids = data["ParticleIDs"][ind]

pos_rel = pos - cat["SubPos"][0,:]
