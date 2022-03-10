#!/bin/env python

import numpy as np
import virgo.mpi.util


def collective_read(dataset, comm):
    """
    Do a parallel collective read of a HDF5 dataset by splitting
    the dataset equally between MPI ranks along its first axis.
    
    File must have been opened in MPI mode.
    """

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
        chunk_size_bytes = 2**30
        chunk_size_elements = chunk_size_bytes // element_size
        local_nr_iterations = num_on_task[comm_rank] // chunk_size_elements
        if num_on_task[comm_rank] % chunk_size_elements > 0:
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
            length = min(nr_left, chunk_size_elements)
            with dataset.collective:
                data[offset_in_mem:offset_in_mem+length,...] = dataset[offset_in_file:offset_in_file+length,...]
            offset_in_mem += length
            offset_in_file += length
            nr_left -= length

    return data


def collective_write(group, name, data, comm):
    """
    Do a parallel collective write of a HDF5 dataset by concatenating
    contributions from MPI ranks along the first axis.
    
    File must have been opened in MPI mode.
    """

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

    # Create the dataset
    dataset = group.create_dataset(name, shape=full_shape, dtype=dtype_rank0)

    # Determine slice to write
    slice_to_write = np.s_[offset_on_task[comm_rank]:offset_on_task[comm_rank]+num_on_task[comm_rank],...]

    # Do a collective write
    with dataset.collective:
        dataset[slice_to_write] = data

    return data


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

        # Determine file indexes
        if file_idx is None:
            if comm_rank == 0:
                filename = filenames % {"i":0}
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

        # Full list of filenames to read
        self.filenames = [filenames % {"i":i} for i in file_idx]
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
            self.file_index = None
            for file_nr, (first_rank, num_ranks) in enumerate(zip(first_rank_on_file, num_ranks_on_file)):
                if comm_rank >= first_rank and comm_rank < first_rank+num_ranks:
                    self.file_index   = file_nr
                    self.rank_in_file = comm_rank - first_rank
                    self.num_ranks    = num_ranks
                    break
            assert self.file_index is not None
            
    def _read_independent(self, datasets, group=None):
        """
        Read and concatenate arrays from multiple files,
        assuming at least one file per MPI rank.
        """

        # Avoid initializing HDF5 (and therefore MPI) until necessary
        import h5py

        rank  = self.comm.Get_rank()
        first = self.first_file_on_rank[rank]
        num   = self.num_files_on_rank[rank]

        data = {name : [] for name in datasets}
        for i in range(first, first+num):
            filename = self.filenames[i]
            with h5py.File(filename, "r") as infile:
                
                # Find HDF5 group to read from
                loc = infile if group is None else infile[group]

                # Read the data, skipping any missing arrays
                for name in data:
                    if name in loc:
                        data[name].append(loc[name][...])
                        
        # Combine data from different files
        for name in data:
            if len(data[name]) > 0:
                data[name] = np.concatenate(data[name])
            else:
                data[name] = None

        return data

    def _read_collective(self, datasets, group=None):
        """
        Read and concatenate arrays from multiple files,
        assuming more MPI ranks than files so each rank
        reads part of a file.
        """

        # Avoid initializing HDF5 (and therefore MPI) until necessary
        import h5py

        # Create communicators for the read:
        # One communicator per file which contains all ranks reading that file.
        comm = self.comm.Split(self.file_index, self.rank_in_file)

        # Dict to store the results
        data = {}

        # Open the file
        filename = self.filenames[self.file_index]
        infile = h5py.File(filename, "r", driver="mpio", comm=comm)

        # Loop over datasets to read
        for name in datasets:

            # Find the dataset
            loc = infile if group is None else infile[group]
            if name not in loc:
                # Dataset doesn't exist in this file
                data[name] = None
            else:
                # Dataset exists, so do a collective read
                data[name] = collective_read(loc[name], comm)

        infile.close()
        comm.Free()

        return data
        
    def read(self, datasets, group=None):

        if self.collective:
            data = self._read_collective(datasets, group)
        else:
            data = self._read_independent(datasets, group)

        # Ensure native endian data: mpi4py doesn't like arrays tagged as big or
        # little endian rather than native.
        for name in data:
            if data[name] is not None:
                data[name] = data[name].astype(data[name].dtype.newbyteorder("="))

        # Create zero length arrays on ranks with no elements.
        # May still return None if all ranks have no elements.
        for name in data:
            data[name] = virgo.mpi.util.replace_none_with_zero_size(data[name], self.comm)

        return data
