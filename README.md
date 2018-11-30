# Examples and read routines for the Virgo Data Centre

## Virgo python module

### Introduction

This is a module which provides facilities for reading the various binary 
formats used to store simulation snapshots, group catalogues and merger trees
in Virgo Consortium simulations.

The interface to these read routines is modelled on h5py. Quantities to read
are specified by name and subsets of arrays can be read in using numpy style array
indexing. This is implemented by memory mapping the input file and creating numpy
arrays which use the appropriate section of the file as their data buffer.

For example, to open a subhalo_tab file from one of the Aquarius simulations:

```
import virgo.formats.subfind_pgadget3 as subfind

fname = "/virgo/simulations/Aquarius/Aq-A/4/groups_1023/subhalo_tab_1023.0"
subtab = subfind.SubTabFile(fname, id_bytes=4)
subtab.sanity_check()
```

Some formats require parameters such as the size of the particle IDs to be specified
before the file can be read. A 'sanity_check()' method is provided which will attempt 
to raise an exception if these parameters are set incorrectly.

Data can then be read from the file as if it were a h5py.File object:
```
subhalo_pos = subtab["SubPos"][...]
subhalo_vel = subtab["SubVel"][...]
```
The .keys() method can be used to see what quantities exist in the file:
```
print subtab.keys()
```

### Layout of the module

  * virgo.formats: contains classes for reading various binary simulation data formats
  * virgo.sims: contains wrappers for the read routines with appropriate default parameters for particular simulations
  * virgo.util: contains utility functions which may be useful when working with simulation data, including
    * the BinaryFile class, which allows for reading binary files using a HDF5-like interface
    * an efficient method for finding matching values in arrays of particle IDs
    * a vectorized python implementation of the peano_hilbert_key function from Gadget
  * virgo.mpi: utilities for working with simulation data using mpi4py, including a reasonably efficient MPI parallel sort

### Reading simulation data

#### Gadget snapshots

The function virgo.formats.gadget_snapshot.open() can be used to read Gadget snapshots
stored in either HDF5 or type 1 binary format. When opening a snapshot file it returns
either a h5py.File object (if the file is in HDF5 format) or a GadgetSnapshotFile
object (if the file is in binary format). In the case of binary files, the precision 
and endian-ness of the file are determined automatically.

Quantities in binary snapshots are accessed using HDF5 style names:
```
import virgo.formats.gadget_snapshot as gs
snap = gs.open("snap_C02_400_1023.0")
boxsize = snap["Header"].attrs["BoxSize"]
pos     = snap["PartType1/Coordinates"][...]
vel     = snap["PartType1/Velocities"][...]
```

#### Subfind output

The following modules contains classes to read subfind and friends of friends output from several versions of Gadget:

  * virgo.formats.subfind_lgadget2 - L-Gadget2, e.g. the Millennium simulation
  * virgo.formats.subfind_lgadget3 - L-Gadget3, e.g. MXXL
  * virgo.formats.subfind_pgadget3 - Millennium2, Aquarius
  * virgo.formats.subfind_gadget4  - The version of Gadget-4 used in COCO (likely not compatible with the current Gadget-4)

These each provide the following classes:

  * SubTabFile - to read subfind group catalogues
  * SubIDsFile - to read the IDs of particles in subfind groups
  * GroupTabFile - to read friends of friends catalogues
  * GroupIDsFile - to read the IDs of particles in friends of friends groups

In some cases extra parameters are required before files can be read:

  * id_bytes - number of bytes used to store a particle ID (4 or 8)
  * float_bytes - number of bytes used to store float quantities in group catalogues (4 or 8)
  * Flags corresponding to Gadget pre-processor macros which affect the output format. These
    should be set to True or False depending on whether the corresponding macro was set.
    * SO_VEL_DISPERSIONS
    * SO_BAR_INFO

See the docstrings associated with each class to determine which parameters are required for which formats.
