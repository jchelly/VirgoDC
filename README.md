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


### Installation

#### Installation of dependencies

The parallel_sort and parallel_hdf5 modules require mpi4py and h5py. On a HPC system
these may need to be built from source to ensure they're linked to the right MPI implementation.

To install mpi4py, ensure that the right mpicc is in your $PATH (e.g. by loading environment modules) and run
```
python -m pip install mpi4py
```

To install h5py, if the right mpicc is in your $PATH and HDF5 is installed at $HDF5_HOME:
```
export CC="mpicc"
export HDF5_MPI="ON"
export HDF5_DIR=${HDF5_HOME}
pip install --no-binary h5py h5py
```

Running the tests requires pytest-mpi:
```
pip install pytest-mpi
```

#### Installing the module

To install the module in your home directory:
```
cd VirgoDC/python
pip install . --user
```

### Layout of the module

  * virgo.formats: contains classes for reading various binary simulation data formats
  * virgo.sims: contains wrappers for the read routines with appropriate default parameters for particular simulations
  * virgo.util: contains utility functions which may be useful when working with simulation data, including
    * the BinaryFile class, which allows for reading binary files using a HDF5-like interface
    * an efficient method for finding matching values in arrays of particle IDs
    * a vectorized python implementation of the peano_hilbert_key function from Gadget
  * virgo.mpi: utilities for working with simulation data using mpi4py, including a reasonably efficient MPI parallel sort and functions for parallel I/O on multi-file simulation output


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

#### Friends-of-Friends and Subfind output

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
In most cases calling the sanity_check() method on the object will raise an exception if any parameters
were set incorrectly.

### MPI Parallel Sorting and Matching Functions

Working with halo finder output often requires matching particle IDs between
the halo finder output files and simulation snapshots.

The module virgo.mpi.parallel_sort contains functions for sorting, matching
and finding unique values in distributed arrays. A distributed array is simply
a numpy array which exists on each MPI rank and is treated as forming part of
a single, large array. Elements in distributed arrays have a notional 'global'
index which, for an array with N elements over all ranks, runs from zero for
the first element on rank zero to N-1 for the last element on the last MPI
rank.

All of these functions are collective and must be executed an all MPI ranks in
the specified communicator `comm`, or `MPI_COMM_WORLD` if the `comm` parameter
is not supplied.

#### Repartitioning a distributed array

```
virgo.mpi.parallel_sort.repartition(arr, ndesired, comm=None)
```
This function returns a copy of the input distributed array `arr` with the
number of elements per MPI rank specified in ndesired, which must be an integer
array with one element per MPI rank.

This can be used to improve memory load balancing if the input array is
unevenly distrubuted. `comm` specifies the MPI communicator to use.
MPI_COMM_WORLD is used if comm is not set.

If `arr` is multidimensional then the repartitioning is done along the first
axis.

#### Fetching specified elements of a distributed array

```
virgo.mpi.parallel_sort.fetch_elements(arr, index, result=None, comm=None)
```
This function returns a new distributed array containing the elements of `arr`
with global indexes specified by `index`. On one MPI rank the result would just
be `arr[index]`. The output can be placed in an existing array using the
`result` parameter.

This function can be used to apply the sorting index returned by
parallel_sort().

If `arr` is multidimensional then the index is taken to refer to the first
axis. I.e. if running with only one MPI rank the function would return
`arr[index,...]`.

#### Sorting

```
virgo.mpi.parallel_sort.parallel_sort(arr, comm=None, return_index=False, verbose=False)
```
This function sorts the supplied array `arr` in place. If `return_index` is
true then it also returns a new distributed array with the global indexes
in the input array of the returned values.

When `return_index` is used this is essentially an MPI parallel version of
np.argsort but with the side effect of sorting the array in place. The index
can be supplied to fetch_elements() to put other arrays into the same order
as the sorted array.

Only one dimensional arrays can be sorted.

This is an attempt to implement the sorting algorithm described in
https://math.mit.edu/~edelman/publications/scalable_parallel.pdf in python,
although there are some differences. For example, the final merging of sorted
array sections is done using a full sort due to the lack of an optimized merge
function in numpy.

#### Finding matching values between two arrays

```
virgo.mpi.parallel_sort.parallel_match(arr1, arr2, arr2_sorted=False, comm=None)
```
For each value in the input distributed array `arr1` this function returns the
global index of an element with the same value in distributed array `arr2`, or
-1 where no match is found.

If `arr2_sorted` is True then the input `arr2` is assumed to be sorted, which
saves some time. Incorrect results will be generated if `arr2_sorted` is true
but `arr2` is not really sorted.

Both input arrays must be one dimensional.

#### Finding unique values

```
virgo.mpi.parallel_sort.parallel_unique(arr, comm=None, arr_sorted=False,
    return_counts=False, repartition_output=False)
```

This function returns a new distributed array which contains the unique values
from the input distributed array arr. If `arr_sorted` is True the input is
assumed to already be sorted, which saves some time.

If `return_counts` is true then it also returns a distributed array with the
number of instances of each unique value which were found.

If `repartition_output` is true then the output arrays are repartitioned to
have approximately equal numbers of elements on each MPI rank.

The input array must be one dimensional.

#### MPI Alltoallv function

```
virgo.mpi.parallel_sort.my_alltoallv(sendbuf, send_count, send_offset,
    recvbuf, recv_count, recv_offset,
    comm=None)
```

Most of the operations in this module use all-to-all communication patterns.
This function provides an all-to-all implementation that avoids problems
with large (>2GB) communications and can handle any numpy type that the mpi4py
`mpi4py.util.dtlib.from_numpy_dtype()` function can translate into an MPI type.

  * `sendbuf` - numpy array with the data to send
  * `send_count` - number of elements to go to each MPI rank
  * `send_offset` - offset of the first element to send to each rank
  * `recvbuf` - numpy array to receive data into
  * `recv_count` - number of elements to go to receive from MPI rank
  * `recv_offset` - offset of the first element to receive from each rank
  * `comm` - specifes the communicator to use (MPI_COMM_WORLD if not set)

#### Tests

The parallel_sort module includes several tests, which can be run with
pytest-mpi:

```
cd VirgoDC/python
mpirun -np 8 python3 -m pytest --with-mpi
```

A larger parallel sort test can be run with
```
mpirun -np 16 python3 -m mpi4py -c "import virgo.mpi.test_parallel_sort as tps ; tps.run_large_parallel_sort(N)"
```
where N is the number of elements per rank to sort. This sorts an array of
random integers and checks that the result is in order and contains the same
number of instances of each value as the input.

### MPI I/O Functions

The module virgo.mpi.parallel_hdf5 contains functions for reading and writing
distributed arrays stored in sets of HDF5 files, using MPI collective I/O
where possible. These can be useful for reading simulation snapshots and halo
finder output.

#### Collective HDF5 Read

```
virgo.mpi.parallel_hdf5.collective_read(dataset, comm)
```
This function carries out a collective read of the dataset `dataset`, which
should be a h5py.Dataset object in a file opened with the 'mpio' driver. It
returns a new distributed array with the data.

Multidimensional arrays are partitioned between MPI ranks along the first
axis.

Reads are chunked if necessary to avoid problems with the underlying MPI
library failing to handle reads of >2GB.

#### Collective HDF5 Write

```
virgo.mpi.parallel_hdf5.collective_write(group, name, data, comm)
```
This function writes the distributed array `data` to the h5py.Group specified
 by the `group` parameter with name `name`.

Multidimensional arrays are assumed to be distributed between MPI ranks along
the first axis.

Note that chunking of large writes is not currently implemented.

#### Multi-file Parallel I/O

##### The MultiFile class

```
virgo.mpi.parallel_hdf5.MultiFile.__init__(self, filenames, file_nr_attr=None,
    file_nr_dataset=None, file_idx=None, comm=None)
```

Simulation codes can often split their output over a variable number of files.
There may be a single large output file, many small files, or something in
between. This class is intended to solve the general problem of carrying out
parallel reads of distributed arrays from N files on M MPI ranks for arbitrary
values of N and M.

The approach is as follows:

  * For N >= M (i.e. at least one file per MPI rank) each MPI rank uses serial
    I/O to read a subset of the files
  * For N < M (i.e. more MPI ranks than files) the MPI ranks are split into
    groups and each group does collective I/O on one file

The class takes the following parameters:
  * `filenames` - a format string to generate the names of files in the set.
    The file number is substituted in as `filenames % {"file_nr" : file_nr}`
  * `file_nr_attr` - a tuple with (HDF5 object name, attribute name) which
    specifies a HDF5 attribute containing the number of files in the set.
    E.g. in a Gadget snapshot use
    `file_nr_attr=("Header","NumFilesPerSnapshot")`.
  * `file_nr_dataset` - the name of a dataset with the number of files in the
     set
  * `file_idx` - an array with the indexes of the files in the set

Exactly one of `file_nr_attr`, `file_nr_dataset` and `file_idx` must be
specified.

##### Reading datasets from a file set

```
virgo.mpi.parallel_hdf5.MultiFile.read(self, datasets, group=None,
    return_file_nr=None)
```
This method reads multiple distributed arrays from the file set. The arrays
are distributed between MPI ranks along the first axis. The parameters are:
  * `datasets` - a list of the names of the datasets to read
  * `group` - the name of the HDF5 group to read datasets from
  * `return_file_nr` - if this is true the output dict contains an extra
    array with the index of the file each element was read from.

Returns a dict containing distributed arrays with one element for each
name in `datasets`. Input datasets should all have the same number of elements
per file.

This can be used to read particles from a snapshot distributed over an
arbitrary number of files, for example.

##### Reading the number of dataset elements per file

```
virgo.mpi.parallel_hdf5.MultiFile.get_elements_per_file(self, name, group=None)
```
This returns the number of elements in each file for the specified dataset
  * `name` - name of the dataset
  * `group` - name of the group containing the dataset

Returns the number of elements per file along the first axis. Note that this
is NOT a distributed array - a copy of the full array is returned on each MPI
rank if this is called collectively.

Can be used with `MultiFile.write()` to write output distributed between files
in the same way as an input file set.

##### Writing datasets to a file set

```
virgo.mpi.parallel_hdf5.MultiFile.write(self, data, elements_per_file,
    filenames, mode, group=None, attrs=None)
```
This writes the supplied distributed arrays to a set of output files with the
specified number of elements per file. The number of output files is the same
as in the input file set used to initialize the class.

  * `data` - a dict containing the distributed arrays to write out. The dict
    keys are used as output dataset names
  * `elements_per_file` - the number of elements along the first axis to write
    to each output file
  * `filenames` - a format string to generate the names of files in the set.
    The file number is substituted in as `filenames % {"file_nr" : file_nr}`
  * `mode` - should be 'r+' to write to existing files or 'w' to create new files
  * `group` - the name of the HDF5 group to write the datasets to
  * `attrs` - a dict containing attributes to add to the datasets, of the form
    `attrs[dataset_name] = (attribute_name, attribute_value)`

The get_elements_per_file() method can be used to get the value of
elements_per_file needed to write output partitioned in the same way as some
input file set.

### MPI Utility Functions

The module virgo.mpi.util contains several other functions which are helpful
for dealing with simulation and halo finder output.

#### Computing particle group membership from Subfind lengths and offsets

```
virgo.mpi.util.group_index_from_length_and_offset(length, offset,
    nr_local_ids, comm=None)
```

Given distributed arrays with the lengths and offsets of particles in a subfind
output, this computes the group index for each particle. The first group is
assigned index zero.
  * `length` - distributed array with the number of particles in each group
  * `offset` - distributed array with the offset to the first particle in each
    group
  * `nr_local_ids` - size of the particle IDs array on this MPI rank. Used to
    determine the size of the output group membership array
  * `comm` - communicator to use. Will use MPI_COMM_WORLD if not specified.

On one MPI rank this would be equivalent to:
```
grnr = np.ones(nr_local_ids, dtype=int)
for i, (l, o) in enumerate(zip(length, offset)):
  grnr[o:o+l] = i
return grnr
```

This can be used in combination with virgo.mpi.parallel_sort.parallel_match()
to find subfind group membership for particles in a simulation snapshot.

#### Allocating zero-sized arrays on ranks with no data

Gadget snapshots typically omit HDF5 datasets which would have zero size (e.g.
if some files in a snapshot happen to have zero star particles). This can be an
issue in parallel programs because MPI ranks which read files with such missing
datasets don't know the type or dimensions of some of the datasets.

```
virgo.mpi.util.replace_none_with_zero_size(arr, comm=None)
```
This takes an input distributed array, `arr`, and on ranks where arr is None
an empty array is returned using type and size information from the other MPI
ranks. The new array will have zero size in the first dimension and the same
size as the other MPI ranks in all other dimensions.

On ranks where the input is not None, the input array is returned.

The array should have the same dtype on all ranks where it is not None.

The intended use of this function is to allow read routines to return None
where datasets do not exist and then this function can be used to retrieve the
missing metadata.
