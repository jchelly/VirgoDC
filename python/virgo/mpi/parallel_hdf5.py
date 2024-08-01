#!/bin/env python

import collections

import numpy as np
import h5py
import virgo.mpi.util

# Default maximum size of I/O operations in bytes.
# This is to avoid MPI issues with buffers >2GB.
BUFFER_SIZE=100*1024*1024


def substitute_file_nr(format_string, file_nr):
    """
    Substitute the file_nr parameter into the supplied format string.
    Allows either % or f-string style formatting.
    """

    str1 = format_string % {"file_nr" : file_nr}
    str2 = format_string.format(file_nr=file_nr)
    if str1 != format_string and str2 != format_string:
        raise RuntimeError(f"Can't determine formatting type for {format_string}")
    elif str1 != format_string:
        return str1
    elif str2 != format_string:
        return str2
    else:
        assert str1==str2
        return str1


def compress_dcpl(dcpl, shape, gzip=None, shuffle=False, chunk=None):
    """
    Convenience function for enabling compression. Enables chunking along
    first dimension if necessary. Returns a new, updated property list and
    leaves the input property list unchanged.

    Note that using romio from openmpi v4 can cause segfaults when writing
    compressed datasets in parallel (see
    https://github.com/open-mpi/ompi/issues/7795).
    """

    dcpl = dcpl.copy()

    # Skip compression and chunking of parallel I/O for HDF5 before 1.12.0
    if h5py.version.hdf5_version_tuple < (1,12,0):
        return dcpl

    # Find total number of elements
    nr_elements = 1
    for n in shape:
        nr_elements *= n

    # Don't chunk or compress zero size datasets
    if nr_elements == 0:
        return

    # Set a default chunk size
    if chunk is None:
        chunk_size = 8*1024*1024
    else:
        chunk_size = int(chunk)

    # Enable chunking if necessary
    if gzip is not None or shuffle:
        if dcpl.get_layout() != h5py.h5d.CHUNKED:
            chunk_shape = list(shape)
            if chunk_shape[0] > chunk_size:
                chunk_shape[0] = chunk_size
            dcpl.set_layout(h5py.h5d.CHUNKED)
            dcpl.set_chunk(tuple(chunk_shape))
        
    # Enable compression
    if shuffle:
        dcpl.set_shuffle()
    if gzip is not None:
        dcpl.set_deflate(gzip)

    return dcpl


class AttributeArray(np.ndarray):
    """
    A numpy ndarray with HDF5 attributes attached
    """
    def __new__(cls, input_array, attrs=None):
        obj = np.asarray(input_array).view(cls)
        obj.attrs = attrs
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.attrs = getattr(obj, 'attrs', None)


def collective_read(dataset, comm, buffer_size=None):
    """
    Do a parallel collective read of a HDF5 dataset by splitting
    the dataset equally between MPI ranks along its first axis.
    
    File must have been opened in MPI mode.
    """

    if buffer_size is None:
        buffer_size = BUFFER_SIZE

    # Avoid initializing HDF5 (and therefore MPI) until necessary
    import h5py
    from mpi4py import MPI

    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Determine how many elements to read on each task
    ntot = dataset.shape[0]
    num_on_task = np.zeros(comm_size, dtype=int)
    num_on_task[:] = ntot // comm_size
    num_on_task[0:ntot % comm_size] += 1
    assert sum(num_on_task) == ntot

    # Determine offsets to read at on each task
    offset_on_task = np.cumsum(num_on_task) - num_on_task

    if ntot < 10*comm_size:
        # If the dataset is small, read on one rank and broadcast
        if comm_rank == 0:
            data = dataset[...]
        else:
            data = None
        data = comm.bcast(data)
        return data[offset_on_task[comm_rank]:offset_on_task[comm_rank]+num_on_task[comm_rank],...]
    else:
        # Otherwise do a collective read
        # Reading >2GB fails, so we may need to chunk the read.
        # Compute amount of local data to read
        element_size = dataset.dtype.itemsize
        for s in dataset.shape[1:]:
            element_size *= s
        local_nr_bytes = num_on_task[comm_rank] * element_size
        # Compute number of iterations this task needs
        buffer_size_elements = buffer_size // element_size
        local_nr_iterations = num_on_task[comm_rank] // buffer_size_elements
        if num_on_task[comm_rank] % buffer_size_elements > 0:
            local_nr_iterations += 1
        # Find maximum number of iterations over all tasks
        global_nr_iterations = comm.allreduce(local_nr_iterations, op=MPI.MAX)
        # Allocate the output array
        shape = list(dataset.shape)
        shape[0] = num_on_task[comm_rank]
        data = np.ndarray(shape, dtype=dataset.dtype)
        # Read the data
        offset_in_mem = 0
        offset_in_file = offset_on_task[comm_rank]
        nr_left = num_on_task[comm_rank]
        for i in range(global_nr_iterations):
            length = min(nr_left, buffer_size_elements)
            with dataset.collective:
                data[offset_in_mem:offset_in_mem+length,...] = dataset[offset_in_file:offset_in_file+length,...]
            offset_in_mem += length
            offset_in_file += length
            nr_left -= length

    return data


def collective_write(group, name, data, comm, buffer_size=None, create_dataset=True,
                     dcpl=None, gzip=None, shuffle=False, chunk=None):
    """
    Do a parallel collective write of a HDF5 dataset by concatenating
    contributions from MPI ranks along the first axis.
    
    File must have been opened in MPI mode.
    """

    if buffer_size is None:
        buffer_size = BUFFER_SIZE

    import h5py
    from mpi4py import MPI

    # Get (or update) dataset creation property list
    if create_dataset:
        if dcpl is None:
            dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        else:
            dcpl = dcpl.copy()

    # Ensure input is a contiguous numpy array
    data = np.ascontiguousarray(data)

    # Find communicator file was opened with
    #comm, info = group.file.id.get_access_plist().get_fapl_mpio()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Determine how many elements to write on each task
    num_on_task = np.asarray(comm.allgather(data.shape[0]))
    ntot = np.sum(num_on_task)

    # Determine offsets at which to write data from each task
    offset_on_task = np.cumsum(num_on_task) - num_on_task

    # Need to have all dimensions but the first the same between ranks
    shape_local = tuple(data.shape[1:])
    shape_rank0 = tuple(comm.bcast(shape_local))
    if shape_local != shape_rank0:
        raise ValueError("Inconsistent data shapes in collective_write()!")

    # Need to have the same data type on all ranks
    dtype_local = data.dtype
    dtype_rank0 = comm.bcast(dtype_local)
    if dtype_local != dtype_rank0:
        raise ValueError("Inconsistent data types in collective_write()!")

    # Find the full shape of the new dataset
    full_shape = [ntot,] + list(shape_local)

    # Create the dataset if necessary
    if create_dataset:
        dspace_id = h5py.h5s.create_simple(tuple(full_shape))
        dtype_id = h5py.h5t.py_create(data.dtype)
        dcpl_compressed = compress_dcpl(dcpl, full_shape, gzip, shuffle, chunk)
        dataset_id = h5py.h5d.create(group.id, name.encode(), dtype_id, dspace_id, dcpl=dcpl_compressed)
        dataset = h5py.Dataset(dataset_id)
    else:
        dataset = group[name]

    # Determine slice to write
    file_offset   = offset_on_task[comm_rank]
    nr_left       = num_on_task[comm_rank]
    memory_offset = 0

    # Determine how many elements we can write per iteration
    element_size = data.dtype.itemsize
    for s in full_shape[1:]:
        element_size *= s
    max_elements = buffer_size // element_size

    # We need to use the low level interface here because the h5py high
    # level interface omits zero sized writes, which causes a hang in
    # collective mode if a rank has nothing to write.
    dset_id = dataset.id
    file_space = dset_id.get_space()
    mem_space = h5py.h5s.create_simple(data.shape)
    prop_list = h5py.h5p.create(h5py.h5p.DATASET_XFER)
    prop_list.set_dxpl_mpio(h5py.h5fd.MPIO_COLLECTIVE)

    # Loop until all elements have been written
    while comm.allreduce(nr_left) > 0:

        # Decide how many elements to write on this rank
        nr_to_write = min(nr_left, max_elements)

        # Select the region in the file
        if nr_to_write > 0:
            start = (file_offset,)+(0,)*(len(data.shape)-1)
            count = (nr_to_write,)+tuple(full_shape[1:])
            file_space.select_hyperslab(start, count)
        else:
            file_space.select_none()

        # Select the region in memory
        if nr_to_write > 0:
            start = (memory_offset,)+(0,)*(len(data.shape)-1)
            count = (nr_to_write,)+tuple(full_shape[1:])
            mem_space.select_hyperslab(start, count)
        else:
            mem_space.select_none()

        # Write the data
        dset_id.write(mem_space, file_space, data, dxpl=prop_list)

        # Advance to next chunk
        file_offset   += nr_to_write
        memory_offset += nr_to_write
        nr_left       -= nr_to_write

    return dataset


def assign_files(nr_files, nr_ranks):
    """
    Assign files to MPI ranks
    """
    files_on_rank = np.zeros(nr_ranks, dtype=int)
    files_on_rank[:] = nr_files // nr_ranks
    remainder = nr_files % nr_ranks
    if remainder > 0:
        step = max(nr_files // (remainder+1), 1)
        for i in range(remainder):
            files_on_rank[(i*step) % nr_ranks] += 1
    assert sum(files_on_rank) == nr_files
    return files_on_rank


class MultiFile:
    """
    Class to read and concatenate arrays from sets of HDF5 files.
    Does parallel reads of N files on M MPI ranks for arbitrary
    N and M.

    filenames - a format string to generate the names of files in the set.
                File number is subbed in as `filenames % {"file_nr" : file_nr}`.
                Alternatively, filenames may be a sequence of strings with
                the names of all files in the set.
    file_nr_attr - a tuple with (HDF5 object name, attribute name) which
                specifies a HDF5 attribute containing the number of files in the set.
                E.g. in a Gadget snapshot use `file_nr_attr=("Header","NumFilesPerSnapshot")`.
    file_nr_dataset - the name of a dataset with the number of files in the set
    file_idx - an array with the indexes of the files in the set
    comm - MPI communicator to use
    """
    def __init__(self, filenames, file_nr_attr=None, file_nr_dataset=None, file_idx=None, comm=None):

        # Avoid initializing HDF5 (and therefore MPI) until necessary
        import h5py

        # MPI communicator to use
        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        self.comm = comm
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()

        # Determine file names and number of files
        if isinstance(filenames, str):
            
            # Filenames is a format string. Need to determine number of files.
            if file_idx is None:
                # File indexes not specified, so read number of files from the first file
                if comm_rank == 0:
                    filename = substitute_file_nr(filenames, 0)
                    with h5py.File(filename, "r") as infile:
                        if file_nr_attr is not None:
                            obj, attr = file_nr_attr
                            nr_files = int(infile[obj].attrs[attr])
                            file_idx = np.arange(nr_files)
                        elif file_nr_dataset is not None:
                            nr_files = int(infile[file_nr_dataset][...])
                            file_idx = np.arange(nr_files)
                        else:
                            raise Exception("Must specify one of file_nr_attr, file_nr_dataset, file_idx")
                else:
                    file_idx = None
                file_idx = comm.bcast(file_idx)
            self.filenames = [substitute_file_nr(filenames, i) for i in file_idx]
            self.all_file_indexes = file_idx

        elif isinstance(filenames, collections.abc.Sequence):

            # Filenames is not a single format string, so interpret as sequence of filenames
            self.filenames = [str(s) for s in filenames]
            nr_files = len(self.filenames)
            self.all_file_indexes = np.arange(nr_files, dtype=int)

        else:
            # Don't know what filenames is in this case!
            raise ValueError("filenames must be string or sequence of strings")

        num_files = len(self.filenames)
        if num_files >= comm_size:
            # More files than ranks, so assign files to ranks
            self.collective = False
            self.num_files_on_rank = assign_files(num_files, comm_size)
            self.first_file_on_rank = np.cumsum(self.num_files_on_rank) - self.num_files_on_rank
        else:
            # More ranks than files, so use collective reading and assign ranks to files.
            self.collective = True
            num_ranks_on_file = assign_files(comm_size, num_files)
            first_rank_on_file = np.cumsum(num_ranks_on_file) - num_ranks_on_file
            # Find which file this rank is assigned to
            self.collective_file_nr = None
            for file_nr, (first_rank, num_ranks) in enumerate(zip(first_rank_on_file, num_ranks_on_file)):
                if comm_rank >= first_rank and comm_rank < first_rank+num_ranks:
                    self.collective_file_nr   = file_nr
                    self.rank_in_file = comm_rank - first_rank
                    self.num_ranks    = num_ranks
                    break
            assert self.collective_file_nr is not None
            
    def _read_independent(self, datasets, group=None, return_file_nr=None, read_attributes=False):
        """
        Read and concatenate arrays from multiple files,
        assuming at least one file per MPI rank.
        """

        # Avoid initializing HDF5 (and therefore MPI) until necessary
        import h5py

        rank  = self.comm.Get_rank()
        first = self.first_file_on_rank[rank]
        num   = self.num_files_on_rank[rank]

        # Dict to store the file index for some datasets
        if return_file_nr is not None:
            file_nr = {name : [] for name in return_file_nr}
        else:
            file_nr = None

        data = {name : [] for name in datasets}
        for i in range(first, first+num):
            filename = self.filenames[i]
            with h5py.File(filename, "r") as infile:
                
                # Find HDF5 group to read from
                if group is None:
                    # No group was specified, so will use the file root group
                    loc = infile
                elif group in infile:
                    # We have a group name and it exists in the file
                    loc = infile[group]
                else:
                    # The specified group does not exist in this file
                    loc = None

                # Read the data, skipping any missing arrays
                if loc is not None:
                    for name in data:
                        if name in loc:

                            # Read the data
                            dataset = loc[name]
                            data[name].append(dataset[...])

                            # Read and store attributes if necessary
                            if read_attributes:
                                attrs = {}
                                for attr_name in dataset.attrs:
                                    attrs[attr_name] = dataset.attrs[attr_name]
                                data[name][-1] = AttributeArray(data[name][-1], attrs=attrs)

                            # Store file number if requested
                            if file_nr is not None and name in file_nr:
                                n = data[name][-1].shape[0]
                                file_nr[name].append(np.ones(n, dtype=int)*self.all_file_indexes[i])

        # Combine data from different files
        for name in data:
            if len(data[name]) > 0:
                if read_attributes:
                    attrs = data[name][0].attrs
                data[name] = np.concatenate(data[name])
                if read_attributes:
                    data[name] = AttributeArray(data[name], attrs=attrs)
            else:
                data[name] = None

        # Combine file indexes
        if file_nr is not None:
            for name in file_nr:
                if len(file_nr[name]) > 0:
                    file_nr[name] = np.concatenate(file_nr[name])
                else:
                    file_nr[name] = None

        return data, file_nr

    def _read_collective(self, datasets, group=None, return_file_nr=None, read_attributes=False):
        """
        Read and concatenate arrays from multiple files,
        assuming more MPI ranks than files so each rank
        reads part of a file.
        """

        # Avoid initializing HDF5 (and therefore MPI) until necessary
        import h5py

        # Create communicators for the read:
        # One communicator per file which contains all ranks reading that file.
        comm = self.comm.Split(self.collective_file_nr, self.rank_in_file)

        # Dict to store the results
        data = {}

        # Dict to store the file index for some datasets
        if return_file_nr is not None:
            file_nr = {}
        else:
            file_nr = None

        # Open the file
        filename = self.filenames[self.collective_file_nr]
        infile = h5py.File(filename, "r", driver="mpio", comm=comm)

        # Find HDF5 group to read from
        if group is None:
            # No group was specified, so will use the file root group
            loc = infile
        elif group in infile:
            # We have a group name and it exists in the file
            loc = infile[group]
        else:
            # The specified group does not exist in this file
            loc = None

        # Loop over datasets to read
        for name in datasets:

            # Find the dataset
            if loc is None or name not in loc:
                # Dataset doesn't exist in this file
                data[name] = None
                if return_file_nr is not None and name in return_file_nr:
                    file_nr[name] = None
            else:
                # Dataset exists, so do a collective read
                dataset = loc[name]
                data[name] = collective_read(dataset, comm)

                # Read and store attributes if necessary
                if read_attributes:
                    attrs = {}
                    for attr_name in dataset.attrs:
                        attrs[attr_name] = dataset.attrs[attr_name]
                    data[name] = AttributeArray(data[name], attrs=attrs)

                # Store file number if needed
                if return_file_nr is not None and name in return_file_nr:
                    n = data[name].shape[0]                    
                    file_nr[name] = np.ones(n, dtype=int)*self.all_file_indexes[self.collective_file_nr]
                    
        infile.close()
        comm.Free()

        return data, file_nr
        
    def read(self, datasets, group=None, return_file_nr=None, read_attributes=False, unpack=False):
        """
        Read and concatenate arrays from a set of one or more files, using
        independent or collective I/O as appropriate.

        Missing datasets are silently skipped. Returns None for datasets where
        no elements are read.

        If datasets is a sequence of strings, returns a dict where the keys are
        the dataset names and the values are numpy arrays. If datasets is a string,
        we read just that dataset and return a numpy array.        

        If unpack=True the results are returned lists rather than dicts. This can
        be used to assign the output arrays directly. E.g.

        pos, vel = mf.read(("Positions", "Velocities"), unpack=True)
        """

        # Check for the case where only one dataset is specified as a string
        if isinstance(datasets, str):
            datasets = (datasets,)
            single_dataset = True
        else:
            single_dataset = False

        if self.collective:
            data, file_nr = self._read_collective(datasets, group, return_file_nr, read_attributes)
        else:
            data, file_nr = self._read_independent(datasets, group, return_file_nr, read_attributes)

        # Ensure native endian data: mpi4py doesn't like arrays tagged as big or
        # little endian rather than native.
        for name in data:
            if data[name] is not None:
                data[name] = data[name].astype(data[name].dtype.newbyteorder("="))

        # Create zero length arrays on ranks with no elements.
        # May still return None if all ranks have no elements.
        for name in data:
            data[name] = virgo.mpi.util.replace_none_with_zero_size(data[name], self.comm)
        if file_nr is not None:
            for name in file_nr:
                file_nr[name] = virgo.mpi.util.replace_none_with_zero_size(file_nr[name], self.comm)

        # Attributes may now be missing from any zero size arrays we created
        if read_attributes:
            for name in data:
                if data[name] is not None:
                    if hasattr(data[name], "attrs"):
                        array_attrs = data[name].attrs
                    else:
                        array_attrs = None
                    array_attrs_all_ranks = self.comm.allgather(array_attrs)
                    if not(hasattr(data[name], "attrs")):
                        for array_attrs in array_attrs_all_ranks:
                            if array_attrs is not None:
                                data[name] = AttributeArray(data[name], attrs=array_attrs)
                                break

        if single_dataset:
            # If we were given a single string instead of a list of names,
            # just return a single array instead of a list of arrays.
            data = data[datasets[0]]
            if file_nr is not None:
                file_nr = file_nr[datasets[0]]
        elif unpack==True:
            # Return lists rather than dicts of arrays if unpack=True
            data = [data[name] for name in datasets]
            if file_nr is not None:
                file_nr = [file_nr[name] for name in datasets]

        if return_file_nr is None:
            return data
        else:
            return data, file_nr

    def get_elements_per_file(self, name, group=None):
        """
        Determine how many elements the specified dataset has in
        each file.
        """

        # Avoid initializing HDF5 (and therefore MPI) until necessary
        import h5py
        
        elements_per_file = {}
        if self.collective:
            # Collective I/O: groups of ranks read a file each
            comm = self.comm.Split(self.collective_file_nr, self.rank_in_file)
            filename = self.filenames[self.collective_file_nr]
            infile = h5py.File(filename, "r", driver="mpio", comm=comm)            
            # Determine group to read from
            if group is None:
                loc = infile
            elif group in infile:
                loc = infile[group]
            else:
                loc = None
            if loc is not None and name in loc:
                ntot = loc[name].shape[0]
                comm_size = comm.Get_size()
                comm_rank = comm.Get_rank()
                num_on_task = np.zeros(comm_size, dtype=int)
                num_on_task[:] = ntot // comm_size
                num_on_task[0:ntot % comm_size] += 1
                assert sum(num_on_task) == ntot
                elements_per_file[self.all_file_indexes[self.collective_file_nr]] = num_on_task[comm_rank]
            else:
                elements_per_file[self.all_file_indexes[self.collective_file_nr]] = 0
            infile.close()
            comm.Free()
        else:
            # Independent I/O: different ranks read different files
            rank  = self.comm.Get_rank()
            first = self.first_file_on_rank[rank]
            num   = self.num_files_on_rank[rank]
            for i in range(first, first+num):
                filename = self.filenames[i]
                with h5py.File(filename, "r") as infile:
                    if group is None:
                        loc = infile
                    elif group in infile:
                        loc = infile[group]
                    else:
                        loc = None
                    if loc is not None and name in loc:
                        elements_per_file[self.all_file_indexes[i]] = loc[name].shape[0]
                    else:
                        elements_per_file[self.all_file_indexes[i]] = 0

        return elements_per_file

    def _write_independent(self, data, elements_per_file, all_filenames, mode, group=None, attrs=None,
                           dcpl=None, gzip=None, shuffle=False, chunk=None):
        """
        Write arrays to multiple files, assuming at least one file per MPI rank.
        """

        # Avoid initializing HDF5 (and therefore MPI) until necessary
        import h5py

        rank  = self.comm.Get_rank()
        first = self.first_file_on_rank[rank]
        num   = self.num_files_on_rank[rank]

        if dcpl is None:
            dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)

        offset = 0
        for i in range(first, first+num):
            filename = all_filenames[i]
            with h5py.File(filename, mode) as outfile:
                
                # Ensure the group exists
                if group is not None:
                    loc = outfile.require_group(group)
                else:
                    loc = outfile

                # Write the data
                length = elements_per_file[self.all_file_indexes[i]]
                for name in data:
                    shape = tuple((length,)+data[name].shape[1:])
                    dcpl_compressed = compress_dcpl(dcpl, shape, gzip, shuffle, chunk)
                    dspace_id = h5py.h5s.create_simple(shape)
                    dtype_id = h5py.h5t.py_create(data[name].dtype)
                    dataset_id = h5py.h5d.create(loc.id, name.encode(), dtype_id, dspace_id, dcpl=dcpl_compressed)
                    dataset = h5py.Dataset(dataset_id)
                    dataset[...] = data[name][offset:offset+length,...]
                    if attrs is not None and name in attrs:
                        for attr_name, attr_val in attrs[name].items():
                            dataset.attrs[attr_name] = attr_val

                offset += length

    def _write_collective(self, data, elements_per_file, all_filenames, mode, group=None, attrs=None,
                          dcpl=None, gzip=None, shuffle=False, chunk=None):
        """
        Write arrays to multiple files in collective mode.
        """

        # Avoid initializing HDF5 (and therefore MPI) until necessary
        import h5py
        comm = self.comm.Split(self.collective_file_nr, self.rank_in_file)

        # Open the file
        filename = all_filenames[self.collective_file_nr]
        outfile = h5py.File(filename, mode, driver="mpio", comm=comm)

        # Ensure the group exists
        if group is not None:
            loc = outfile.require_group(group)
        else:
            loc = outfile
        
        # Write the data
        for name in data:
            length = elements_per_file[self.all_file_indexes[self.collective_file_nr]]
            assert length == data[name].shape[0]
            length_tot = comm.allreduce(length)
            dataset = collective_write(loc, name, data[name], comm, dcpl=dcpl,
                                       gzip=gzip, shuffle=shuffle, chunk=chunk)
            if attrs is not None and name in attrs:
                for attr_name, attr_val in attrs[name].items():
                    dataset.attrs[attr_name] = attr_val
         
        outfile.close()
        comm.Free()

    def write(self, data, elements_per_file, filenames, mode, group=None, attrs=None, dcpl=None,
              gzip=None, shuffle=False, chunk=None):
        """
        Write out the supplied datasets with the same layout as the
        input. Use mode parameter to choose whether to create new
        files or modify existing files.
        """

        # Determine if we have a list of filenames or a single format string
        if isinstance(filenames, str):
            # filenames is a format string. Substitute in the file indexes.
            all_filenames = [substitute_file_nr(filenames, i) for i in self.all_file_indexes]
        elif isinstance(filenames, collections.abc.Sequence):
            # filenames is a list of filenames
            all_filenames = [str(fn) for fn in filenames]
            # Must have the same number of output files as input files
            if len(all_filenames) != len(self.all_file_indexes):
                raise ValueError("wrong number of filenames provided")
        else:
            raise ValueError("filenames must be string or sequence of strings")

        if self.collective:
            # Collective mode
            self._write_collective(data, elements_per_file, all_filenames, mode, group, attrs, dcpl,
                                   gzip, shuffle, chunk)
        else:
            # Independent mode
            self._write_independent(data, elements_per_file, all_filenames, mode, group, attrs, dcpl,
                                    gzip, shuffle, chunk)
            
