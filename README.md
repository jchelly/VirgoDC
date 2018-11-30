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
before the file can be read. A 'sanity_check()' method is provided which attempts 
to catch the case where these parameters are set incorrectly.

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



