#!/bin/env python
#
# Read in the subfind catalogue for a Millennium snapshot
# and write it to stdout as text.
#

import virgo.sims.millennium as mill

# Snapshot number to read
isnap = 63

print "Parent halo, Number of particles, x, y, z, vx, vy, vz"

# Loop over subtab files in this snapshot
for ifile in range(512):

    # Open this file
    fname = "/virgo/data/Millennium/postproc_%03d/sub_tab_%03d.%d" % (isnap, isnap, ifile)
    subtab = mill.SubTabFile(fname)

    #
    # Read some subhalo properties from this file
    # Full list of property names can be obtained with subtab.keys().
    #
    nsub       = subtab["Nsubhalos"][...] # No. of subhalos in this file
    parenthalo = subtab["SubParentHalo"][...]
    len        = subtab["SubLen"][...]    # No. of particles in subhalo    
    pos        = subtab["SubPos"][...]    # Potential minimum coordinates of each subhalo (comoving Mpc.h)
    vel        = subtab["SubVel"][...]    # Velocity of each subhalo (km/sec peculiar)

    for i in range(nsub):
        print "%d, %d, %14.6f, %14.6f, %14.6f, %14.6f, %14.6f, %14.6f" % (parenthalo[i], len[i], 
                                                                          pos[i,0], pos[i,1], pos[i,2],
                                                                          vel[i,0], vel[i,1], vel[i,2])
