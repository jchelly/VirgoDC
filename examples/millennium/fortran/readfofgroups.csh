#!/bin/csh
#
# Parameters for cdml batch queue system 
#$ -N readfofgroups
#$ -cwd
#$ -l sh
#
# Read the friends of friends groups for one millennium output file,
# in this case file 0 at snapshot 63 (z=0).
#
readfofgroupsexample.exe << EOF
0
MILLENNIUM
snap_millennium
63
/data/rw1/jch/fof_group_particles
EOF
