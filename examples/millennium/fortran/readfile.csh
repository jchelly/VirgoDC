#!/bin/csh
#
# Read file 431 from snapshot 58 and output particle coordinates
#
./readfileexample << EOF
/virgo/data/Millennium/
snap_millennium
58
431
./particles_058.431.txt
EOF
