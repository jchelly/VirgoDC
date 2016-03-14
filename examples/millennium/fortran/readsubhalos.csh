#!/bin/csh
#
# Read the halo and subhalo properties from snapshot 63, file 0.
#
readsubhalosexample.exe << FIN
0
MILLENNIUM
snap_millennium
63
./output_fofhalos
./output_subhalos
FIN
