#!/bin/csh
#
# Read all particles in a cube with x,y,z coordinates between
# 70 and 80Mpc/h from snapshot 60
#
./readregionexample << FIN
70.0,80.0
70.0,80.0
70.0,80.0
500.0
/virgo/data/Millennium/
snap_millennium
60
16777216
512
./output_region.txt
FIN
