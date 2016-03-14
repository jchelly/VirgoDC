#!/bin/csh
#
# Read the SubFind groups from file 12 of snapshot 60
# and output the particle coordinates and group membership
#
./readgroupsexample << FIN
12
/virgo/data/Millennium/
snap_millennium
60
16777216
512
./output_groups
FIN
