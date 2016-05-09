#!/bin/env python

import os
import numpy as np
from collections import OrderedDict, Mapping
import mmap
import weakref
import sys
import hashlib

try:
    import h5py
except ImportError:
    have_hdf5 = False
else:
    have_hdf5 = True


def big_or_little(endian):
    """Ensure an endian specifier is < or > and not ="""
    if endian == "=":
        if sys.byteorder == "little":
            endian = "<"
        else:
            endian = ">"
    return endian


def need_byteswap(data_endian, dtype):
    """
    Return True if the assumed byte order of the data given by data_endian
    is not the same as that of the type dtype.

    Some types don't have byte order, in which case we return False.
    """
    if dtype.byteorder == "|":
        return False # Byte order not applicable for this type
    if data_endian not in ("<",">","="):
        raise ValueError("Invalid byte order value - must be '<', '>', or '='")
    data_endian   = big_or_little(data_endian)
    memory_endian = big_or_little(dtype.byteorder)
    return memory_endian != data_endian


def _split_path(path):
    """
    Split a path at the first slash (excluding leading slashes), and return
    a tuple (head,tail) where head is the part before the slash and tail is
    the part after it. If there is no slash then head will be None and tail
    will be the full string. Leading slashes are stripped before the split
    is done.

    This is not the same as os.path.split(), which splits at the last path
    separator.
    """
    components = path.strip("/").split("/",1)
    if len(components) == 1:
        return (None, components[0])
    else:
        return components


class MemoryMappedFile:
    """
    Class representing a memory map of a complete file. May be
    closed, in which case it's no longer valid to use.
    """
    def __init__(self, fname):
        self.closed = False
        self.fname  = fname
        self.file   = open(fname, "r")
        self.mmap   = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
    def __del__(self):
        self.close()
    def close(self):
        self.closed = True
        if hasattr(self, "mmap"):
            if self.mmap is not None:
                self.mmap.close()
                self.mmap = None
        if hasattr(self, "file"):
            if self.file is not None:
                self.file.close()
                self.file = None
    def __repr__(self):
        return ('MemoryMappedFile("%s")' % (self.fname,))


class BinaryAttrs(Mapping):
    """
    Class to store attributes of a group or dataset. This is really
    a dictionary containing BinaryDataset objects, but indexing triggers
    a read of the underlying data rather than just returning the object.
    """
    def __init__(self):
        self._items = OrderedDict()

    def __getitem__(self, key):
        return self._items[key][()]

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        for key in self._items.keys():
            yield key


class BinaryGroup(Mapping):
    """
    Class used to mimic h5py style group access in a binary file
    """
    def __init__(self, name):
        self.attrs    = BinaryAttrs()
        self.datasets = OrderedDict()
        self.groups   = OrderedDict()
        self.name     = name

    def __getitem__(self, key):
        head, tail = _split_path(key)
        if head is None:
            if tail in self.groups:
                return self.groups[tail]
            else:
                return self.datasets[tail]
        else:
            return self.groups[head][tail]

    def __len__(self):
        return len(self.datasets) + len(self.groups)

    def __iter__(self):
        for key in self.datasets.keys() + self.groups.keys():
            yield key
        
    def _create_group(self, name):
        if name in self.datasets:
            raise KeyError("Creating group with same name as dataset!")
        elif name in self.groups:
            raise KeyError("Duplicate group name!")
        else:
            self.groups[name] = BinaryGroup(name)

    def _add_block(self, name, dtype, shape, offset, endian, fname, 
                  is_attr):
        if endian not in ("<",">","="):
            raise ValueError("Invalid endian specification!")
        head, tail = _split_path(name)
        if head is None:
            # Path refers to a dataset or attribute in this group
            new_block = BinaryDataset(fname, dtype, offset, shape, tail)
            if is_attr:
                if tail in self.attrs:
                    raise KeyError("Duplicate attribute name!")
                self.attrs._items[tail] = new_block
            else:
                if tail in self.datasets:
                    raise KeyError("Duplicate dataset name!")
                if tail in self.groups:
                    raise KeyError("Creating dataset with same name as group!")
                self.datasets[tail] = new_block
            new_block.endian = endian
        else:
            # Path refers to a dataset or attribute in a sub-group
            if is_attr and head in self.datasets:
                # We're adding attributes to a dataset in this group
                if tail in self.datasets[head].attrs:
                    raise KeyError("Duplicate attribute name!")
                new_block = BinaryDataset(fname, dtype, offset, shape, tail)
                self.datasets[head].attrs._items[tail] = new_block
            else:
                # We're adding something to a sub-group, so ensure sub-group exists
                if head not in self.groups:
                    self._create_group(head)
                # Create dataset or attribute in sub group
                new_block = self.groups[head]._add_block(tail, dtype, shape, 
                                                         offset, endian, fname, 
                                                         is_attr)
        return new_block

    def __repr__(self):
        return ('<BinaryGroup "%s">' % (self.name,))

    def write_hdf5(self, filename=None, h5group=None, mode="w-"):
        """Write the contents of the file to a new HDF5 file"""

        # Create the output file if necessary
        if h5group is None:
            h5group = h5py.File(filename, mode)
            created = True
        else:
            created = False

        # Loop over datasets in this group and write them out
        for dataset_name in self.datasets.keys():
            h5group[dataset_name] = self.datasets[dataset_name][...]
            for attr_name in self.datasets[dataset_name].attrs.keys():
                h5group[dataset_name].attrs[attr_name] = self.datasets[dataset_name].attrs[attr_name]

        # Loop over groups in this group and create in output file
        for group_name in self.groups.keys():
            new_group = h5group.create_group(group_name)
            for attr_name in self.groups[group_name].attrs.keys():
                new_group.attrs[attr_name] = self.groups[group_name].attrs[attr_name]
            # Recursively write sub-groups
            self.groups[group_name].write_hdf5(h5group=new_group)

        # Close the file if we created a new one
        if created:
            h5group.close()


class BinaryDataset:
    """
    Class representing a block of data in a binary file which is to
    be interpreted as a numpy array of specified dtype and shape.
    offset is offset into the file in bytes.

    Note that dtype is the required type in memory. If this type is of
    different byte order to the assumed byte order of the data in
    the file, conversion will be done.

    BinaryDatasets can be indexed like numpy arrays (because we just
    create a memory mapped array and pass the slice object to numpy)
    """

    max_files  = 32 # Maximum number of files to keep open at once
    file_cache = OrderedDict()

    def __init__(self, fname, dtype, offset, shape, name):
        self.file   = None
        self.fname  = fname
        self.dtype  = np.dtype(dtype)
        self.offset = offset
        self.shape  = tuple([int(i) for i in shape]) # Ensure we have tuple of ints
        self.name   = name
        self.size  = 1
        self.endian = "="
        self.attrs  = BinaryAttrs()
        for s in shape:
            self.size *= s
        self.nbytes = self.dtype.itemsize * self.size

    def _mmap_file(self):
        """
        Ensure we have a valid MemoryMappedFile object for the current file.
        May use an existing object from the cache if the file has already
        been opened.

        This is to avoid repeatedly opening/closing the same file while also
        avoiding running out of file descriptors.
        """

        # If file is already mapped, there's nothing to do
        if self.file is not None and not(self.file.closed):
            return

        # Check if file is already open
        if self.fname in BinaryDataset.file_cache:
            cached_file = BinaryDataset.file_cache[self.fname]()
            if cached_file is None or cached_file.closed:
                # File was already closed or deallocated, so remove cache entry
                del BinaryDataset.file_cache[self.fname]
            else:
                # File is open already, so we can use it
                self.file = cached_file
                return

        # If there are too many files open, remove the oldest first
        while len(BinaryDataset.file_cache) >= BinaryDataset.max_files:
            oldest_file = BinaryDataset.file_cache.popitem(last=False)[1]()
            if oldest_file is not None:
                oldest_file.close()

        # Can now open a new file and add it to the cache
        new_file = MemoryMappedFile(self.fname)
        BinaryDataset.file_cache[self.fname] = weakref.ref(new_file)
        self.file = new_file

    def __getitem__(self, key):
        # Ensure file is open and memory mapped
        self._mmap_file()
        # Check that array to read doesn't fall outside the memory map
        if self.offset < 0:
            raise IOError("Found negative offset when reading binary file!")
        file_size  = self.file.mmap.size()
        if self.offset+self.nbytes > file_size:
            raise IOError("Attempt to read past end of file!")
        # Create memory mapped numpy array
        array = np.ndarray(shape=self.shape, dtype=self.dtype,
                           buffer=self.file.mmap,
                           offset=self.offset)
        # Extract the requested part of the array, copying to make
        # sure we don't return a view of the memory map because
        # we might need to close the file.
        if need_byteswap(self.endian, self.dtype):
            array = array[key].byteswap()
        else:
            array = array[key].copy()
        # Return the result
        return array

    def __array__(self, dtype=None):
        if dtype is None:
            return self[...]
        else:
            return self[...].astype(dtype)

    def __repr__(self):
        return ('<BinaryDataset "%s": shape %s, type %s>' % 
                (self.name, str(self.shape), str(self.dtype)))

    def __len__(self):
        if len(self.shape) >= 1:
            return self.shape[0]
        else:
            raise TypeError("Can't return len of scalar dataset!")

    def len(self):
        if len(self.shape) >= 1:
            return self.shape[0]
        else:
            raise TypeError("Can't return len of scalar dataset!")
    
    def __getattr__(self, name):
        if name == "value":
            return self[()]
        else:
            raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))

    def __iter__(self):
        if len(self.shape) >= 1:
            for i in xrange(self.shape[0]):
                yield self[i,...]
        else:
            raise TypeError("Can't iterate a scalar dataset!")
        


class BinaryFile(BinaryGroup):
    """
    Class representing a binary file containing blocks of data.
    Use calls to add_dataset and add_attribute to describe the 
    data in a particular file.

    Once blocks have been defined a BinaryFile can be accessed 
    as a dictionary, where the keys are block names and the 
    values are BinaryDataset objects.

    Calling set_endian() sets assumed byte order for all existing
    blocks and sets the default for new blocks. Allowed values are:

    '=' - data are native endian
    '<' - data are little endian
    '>' - data are big endian

    Alternatively can call enable_byteswap(False) to specify native
    endian data or enable_byteswap(True) to specify that data are 
    opposite endian to this system.
    """
    def __init__(self, fname):
        BinaryGroup.__init__(self, "")
        self.fname = fname
        self.offset = 0
        self.endian = "="
        self.all_blocks = []
        self.fortran_record = None

    def _add_block(self, name, dtype, shape, is_attr):

        # Add the new data block
        new_block = BinaryGroup._add_block(self, name, dtype, shape, 
                                           self.offset, self.endian, 
                                           self.fname, is_attr)

        # Increment offset into file
        self.offset += new_block.nbytes

        # Keep list of all blocks in the file so we can toggle byteswapping
        self.all_blocks.append(new_block)

    def add_dataset(self, name, dtype, shape=()):
        self._add_block(name, dtype, shape, is_attr=False)

    def add_attribute(self, name, dtype, shape=()):
        self._add_block(name, dtype, shape, is_attr=True)

    def skip_bytes(self, n):
        self.offset += n

    def read_and_skip(self, dtype, shape=()):
        tmp = BinaryDataset(self.fname, dtype, self.offset, shape, None)
        tmp.endian = self.endian
        self.offset += tmp.nbytes
        self.all_blocks.append(tmp)
        return tmp[()]

    def set_endian(self, endian):
        """
        Set the assumed byte order of data in the file. Allowed values:

        < : little endian
        > : big endian
        = : same endian as this system
        """
        if endian not in ("<",">","="):
            raise ValueError("Invalid endian specification!")
        self.endian = endian
        for block in self.all_blocks:
            block.endian = endian

    def enable_byteswap(self, byteswap):
        """
        Specify whether the data in the file needs byte swapping on this system:
        
        byteswap = True  - data is opposite endian to this system
        byteswap = False - data is same endian as this system
        """
        if sys.byteorder == "little":
            if byteswap:
                endian = ">"
            else:
                endian = "<"
        else:
            if byteswap:
                endian = "<"
            else:
                endian = ">"    
        self.set_endian(endian)

    def toggle_byteswap(self):
        """Reverse the current byte swapping setting"""
        if self.endian == "<":
            self.endian = ">"
        elif self.endian == ">":
            self.endian = "<"
        elif self.endian == "=":
            self.enable_byteswap(True)

    def __repr__(self):
        return ('BinaryFile("%s")' % (self.fname,))

    def write_binary(self, fname, mode="wb-", endian=None):
        """
        Write all blocks in this file to a new file. Output will be
        in byte order given by endian parameter, or same byte order
        as data types in add_dataset calls if endian=None.

        Note: this only works if the endian flag has been set such
        that the data will be read correctly into memory.
        """
        
        # Get list of blocks sorted by offset
        blocks = sorted(self.all_blocks, key=lambda x: x.offset)
            
        # Create the output file and write data blocks
        outfile = open(fname, mode)
        for block in blocks:
            # Read this data block
            data = block[...]
            # Byte swap if necessary
            if endian is not None:
                data = data.astype(data.dtype.newbyteorder(endian))
            # Write the block
            outfile.seek(block.offset)
            outfile.write(data.data)
        outfile.close()

    def hash(self, hash_type="sha1"):
        """
        Compute hash of the data in this file.
        Blocks are converted to little endian first so result should be
        independent of byte order, assuming file endian flag is set
        correctly.

        If every byte in the file belongs to exactly one block and
        byte swapping is not required the output from this should be
        exactly the same as the output of the sha1sum (or similar) command.
        """
        
        # Only allow hashing if all bytes in the file belong to a block.
        # This is so that we don't wrongly conclude that files are identical
        # by only comparing a subset of the bytes.
        if not(self.all_bytes_used()):
            raise IOError("All bytes must belong to a dataset before a file can be hashed!")

        # Get list of blocks sorted by offset
        blocks = sorted(self.all_blocks, key=lambda x: x.offset)
            
        # Read data and compute hash
        file_hash = hashlib.new(hash_type)
        for block in blocks:
            data = block[...]
            data = data.astype(data.dtype.newbyteorder("<"))
            file_hash.update(data)
        return file_hash.hexdigest()

    def start_fortran_record(self, auto_byteswap=False):
        if self.fortran_record is not None:
            raise IOError("Tried to start new fortran record without ending previous record!")
        self.fortran_record = (self.read_and_skip(np.int32), self.offset, auto_byteswap)

    def end_fortran_record(self):
        if self.fortran_record is None:
            raise IOError("Tried to read fortran record while not in a record!")
        irec_start, start_offset, auto_byteswap = self.fortran_record
        end_offset = self.offset
        irec_end = self.read_and_skip(np.int32)
        # Check start and end markers agree
        if irec_start != irec_end:
            raise IOError("Start and end of record markers don't match!")
        # Auto byteswapping is only reliable for short records
        if auto_byteswap and (end_offset - start_offset) > 65535:
            raise IOError("Automatic byteswapping can only be used for records < 64kbytes")
        # Check if we need to switch the byte swapping setting
        if (end_offset - start_offset == irec_start.byteswap() and 
            end_offset - start_offset != irec_start and auto_byteswap):
            self.toggle_byteswap()
            irec_start = irec_start.byteswap()
        # Check length agrees with record markers
        if end_offset - start_offset != irec_start:
            raise IOError("Record markers do not agree with length of record!")
        self.fortran_record = None

    def all_bytes_used(self):
        """
        Check that every byte in the file belongs to exactly one
        dataset. Return True if this is the case, False otherwise.
        Can be used for sanity checks on read routines.
        """
        
        # Get lengths and offsets of blocks in the order they appear in the file
        blocks  = sorted(self.all_blocks, key=lambda x: x.offset)
        offsets = np.asarray([b.offset for b in blocks], dtype=np.int64)
        lengths = np.asarray([b.nbytes for b in blocks], dtype=np.int64)

        # First block (if any) must start at offset zero
        if len(blocks) > 0 and offsets[0] != 0:
            return False
        
        # Each block must start immediately after the previous block
        if len(blocks) > 1:
            if np.any(offsets[1:] != offsets[:-1] + lengths[:-1]):
                return False

        # Sum of block sizes must equal file size
        file_size   = os.stat(self.fname).st_size
        blocks_size = np.sum(lengths)
        if file_size != blocks_size:
            return False

        # If we get to here, every byte belongs to exactly one dataset
        return True
