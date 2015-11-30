#!/bin/env python
#
# Classes to read binary SubFind output from Gadget-4
# (at least the version used for COCO)
#

import numpy as np
from virgo.util.read_binary import BinaryFile
from virgo.util.exceptions  import SanityCheckFailedException
from virgo.util.read_multi import read_multi
from virgo.formats.gadget_snapshot import GadgetSnapshotFile

class SubTabFile(BinaryFile):
    """
    Class for reading sub_tab files written by Gadget-4
    """
    def add_record(self, name, dtype, shape=()):
        """Add a Fortran record containing a single dataset to the file"""
        self.start_fortran_record()
        self.add_dataset(name, dtype, shape)
        self.end_fortran_record()

    def __init__(self, fname, id_bytes=8, *args):
        BinaryFile.__init__(self, fname, *args)

        # We need to know the data types used for particle IDs
        if id_bytes == 4:
            self.id_type = np.uint32
        elif id_bytes == 8:
            self.id_type = np.uint64
        else:
            raise ValueError("id_bytes must be 4 or 8")

        # Read file header and determine endian-ness
        self.start_fortran_record(auto_byteswap=True)
        self.add_dataset("Ngroups",       np.int32)
        self.add_dataset("Nsubgroups",    np.int32)
        self.add_dataset("Nids",          np.int32)
        self.add_dataset("TotNgroups",    np.int32)
        self.add_dataset("TotNsubgroups", np.int32)
        self.add_dataset("TotNids",       np.int32)
        self.add_dataset("NTask",         np.int32)
        self.add_dataset("padding1",      np.int32)
        self.add_dataset("Time",        np.float64)
        self.add_dataset("Redshift",    np.float64)
        self.add_dataset("HubbleParam", np.float64)
        self.add_dataset("BoxSize",     np.float64)
        self.add_dataset("Omega0",      np.float64)
        self.add_dataset("OmegaLambda", np.float64)
        self.add_dataset("flag_dp",     np.int32)
        self.add_dataset("padding2",      np.int32)
        self.end_fortran_record()

        # Check if this output uses double precision floats
        flag_dp = self["flag_dp"][...]
        if flag_dp == 0:
            self.float_type = np.float32
        else:
            self.float_type = np.float64

        # Data blocks for FoF groups
        # These are Fortran records which are only present if ngroups > 0.
        ngroups = self["Ngroups"][...]
        if ngroups > 0:
            self.add_record("GroupLen",          np.int32,        (ngroups,))
            self.add_record("GroupMass",         self.float_type, (ngroups,))
            self.add_record("GroupPos",          self.float_type, (ngroups,3))
            self.add_record("GroupVel",          self.float_type, (ngroups,3))
            self.add_record("GroupLenType",      np.int32,        (ngroups,6))
            self.add_record("GroupMassType",     self.float_type, (ngroups,6))
            self.add_record("Halo_M_Mean200",    self.float_type, (ngroups,))
            self.add_record("Halo_R_Mean200",    self.float_type, (ngroups,))
            self.add_record("Halo_M_Crit200",    self.float_type, (ngroups,))
            self.add_record("Halo_R_Crit200",    self.float_type, (ngroups,))
            self.add_record("Halo_M_TopHat200",  self.float_type, (ngroups,))
            self.add_record("Halo_R_TopHat200",  self.float_type, (ngroups,))
            self.add_record("Nsubs",             np.int32,        (ngroups,))
            self.add_record("FirstSub",          np.int32,        (ngroups,))
            
        # Data blocks for Subfind groups
        # These are Fortran records which are only present if nsubgroups > 0.
        nsubgroups = self["Nsubgroups"][...]
        if nsubgroups > 0:
            self.add_record("SubLen",             np.int32,        (nsubgroups,))
            self.add_record("SubMass",            self.float_type, (nsubgroups,))
            self.add_record("SubPos",             self.float_type, (nsubgroups,3))
            self.add_record("SubVel",             self.float_type, (nsubgroups,3))
            self.add_record("SubLenType",         np.int32,        (nsubgroups,6))
            self.add_record("SubMassType",        self.float_type, (nsubgroups,6))
            self.add_record("SubCofM",            self.float_type, (nsubgroups,3))
            self.add_record("SubSpin",            self.float_type, (nsubgroups,3))
            self.add_record("SubVelDisp",         self.float_type, (nsubgroups,))
            self.add_record("SubVmax",            self.float_type, (nsubgroups,))
            self.add_record("SubRVmax",           self.float_type, (nsubgroups,))
            self.add_record("SubHalfMassRad",     self.float_type, (nsubgroups,))
            self.add_record("SubHalfMassRadType", self.float_type, (nsubgroups,6))
            self.add_record("SubMassInRad",       self.float_type, (nsubgroups,))
            self.add_record("SubMassInRadType",   self.float_type, (nsubgroups,6))
            self.add_record("SubMostBoundID",     self.id_type,    (nsubgroups,))
            self.add_record("SubGrNr",            np.int32,        (nsubgroups,))
            self.add_record("SubParent",          np.int32,        (nsubgroups,))


class GroupCatalogue(Mapping):
    """
    Class for reading the complete group catalogue for
    a snapshot into memory.

    This class acts as a dictionary where the keys are dataset
    names and the values are numpy arrays with the data.
    """
    def __init__(self, basedir, isnap, id_bytes=8, datasets=None):

        # Default datasets to read
        if datasets is None:
            datasets = ["GroupLen", "GroupMass", "GroupPos", "GroupVel", "GroupLenType", "GroupMassType", 
                        "Halo_M_Mean200",   "Halo_R_Mean200", 
                        "Halo_M_Crit200",   "Halo_R_Crit200", 
                        "Halo_M_TopHat200", "Halo_R_TopHat200", 
                        "Nsubs", "FirstSub", "SubLen", "SubMass", "SubPos", "SubVel", 
                        "SubLenType", "SubMassType", "SubCofM", "SubSpin", "SubVelDisp", 
                        "SubVmax", "SubRVmax", "SubHalfMassRad", "SubHalfMassRadType", 
                        "SubMassInRad", "SubMassInRadType", "SubMostBoundID", "SubGrNr", "SubParent"]

        # Construct format string for file names
        fname_fmt = ("%s/groups_%03d/subhalo_tab_%03d" % (basedir, isnap, isnap)) + ".%(i)d"

        # Get number of files
        f = SubTabFile(fname_fmt % {"i":0}, id_bytes=id_bytes)
        nfiles = f["NTask"][...]
        del f
        
        # Read the catalogue data
        self._items = read_multi(SubTabFile, fname_fmt, np.arange(nfiles), datasets, 
                                 id_bytes=id_bytes)

    def __getitem__(self, key):
        return self._items[key]

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        for key in self._items.keys():
            yield key


class GroupOrderedSnapshot():
    """
    Class for reading fof groups and subhalos from a snapshot
    where particles have been sorted by group membership.
    """
    def __init__(self, basedir, basename, isnap, id_bytes=8):
        """
        Open a new snapshot

        basedir:  name of directory with snapdir_??? and groups_??? subdirs
        basename: snapshot file name before the last underscore (e.g. snap_millennium)
        isnap:    which snapshot to read
        """
        
        # Store file name info
        self.basedir    = basedir
        self.basename   = basename
        self.isnap      = isnap
        self.id_bytes   = id_bytes

        # Read group lengths from all files
        self.foflentype = []
        ifile  = 0
        nfiles = 1
        while ifile < nfiles:

            # Open file and get number of files
            sub = self.open_subtab_file(ifile)
            if ifile == 0:
                nfiles = sub["NTask"][...]

            # Read group lengths
            nfof = sub["Ngroups"][...]
            if nfof > 0:
                self.foflentype.append(sub["GroupLenType"][...])
        
            # Store number of groups in each file
            if ifile == 0:
                self.nfof_file = -np.ones(nfiles, dtype=np.int32)
            self.nfof_file[ifile] = nfof

            ifile += 1
            
        self.foflentype = np.concatenate(self.foflentype, axis=0)

        # Calculate offset to each fof group for each particle type
        self.fof_offset = np.cumsum(self.foflentype, dtype=np.int64, axis=0) - self.foflentype

        # Find number of snapshot files in this snapshot
        snap = self.open_snap_file(0)
        self.num_snap_files = snap["Header"].attrs["NumFilesPerSnapshot"]
        self.npart_file = -np.ones((self.num_snap_files,6), dtype=np.int64)

        # Calculate index of first fof group in each file
        self.first_fof_in_file = np.cumsum(self.nfof_file) - self.nfof_file

    def open_snap_file(self, ifile):
        """
        Open the specified snapshot file
        """
        fname = "%s/snapdir_%03d/%s_%03d.%d" % (self.basedir, self.isnap, 
                                                self.basename, self.isnap, ifile)
        return GadgetSnapshotFile(fname) 
    
    def open_subtab_file(self, ifile):
        """
        Open the specified subhalo tab file
        """
        fname = "%s/groups_%03d/fof_subhalo_tab_%03d.%d" % (self.basedir, self.isnap, self.isnap, ifile)
        return SubTabFile(fname, id_bytes=self.id_bytes) 

    def read_fof_group(self, grnr):
        """
        Read pos, vel, type for particles in the specified FoF group

        grnr can either be a single integer giving the position of the group
        in the full catalogue, or a two element sequence with the file number
        and position relative to the start of that file

        Returns a list with one element for each particle type.
        Each element is a dictionary containing the Coordinates, Velocities
        and ParticleIDs arrays.
        """

        try:
            ifof = int(grnr)
        except TypeError:
            filenum, fofnum = grnr
            ifof = self.first_fof_in_file[filenum] + fofnum

        # Output will be a list with one entry per particle type.
        # Each list element will be a dictionary with Coordinates, ParticleIDs etc.
        result = [{} for _ in range(6)]

        # Determine which particles we need to read for each type
        first_in_fof = self.fof_offset[ifof,:]
        last_in_fof  = first_in_fof + self.foflentype[ifof,:]
        
        # Now loop over snapshot files and read those we need
        first_in_snap = np.zeros(6, dtype=np.int64)
        for ifile in range(self.num_snap_files):
            
            snap = None

            # Read number of particles in file if necessary
            if self.npart_file[ifile,0] == -1:
                snap = self.open_snap_file(ifile)
                self.npart_file[ifile,:] = snap["Header"].attrs["NumPart_ThisFile"][:]

            # Check if any particles we need are in this file
            last_in_snap = first_in_snap + self.npart_file[ifile,:]

            # Loop over quantities to read
            for dataset in ("Coordinates","Velocities","ParticleIDs"):
                    
                # Loop over particle types
                for itype in range(6):

                    # Check if snapshot has particles of this type
                    if last_in_snap[itype] >= first_in_snap[itype]:

                        # Find range of particles to read from this file
                        first_to_read = first_in_fof[itype] - first_in_snap[itype]
                        last_to_read  = last_in_fof[itype]  - first_in_snap[itype]
                        if first_to_read < 0:
                            first_to_read = 0
                        if last_to_read >= self.npart_file[ifile,itype]:
                            last_to_read = self.npart_file[ifile,itype] - 1

                        # Read the particles, if there are any in this file
                        if last_to_read >= first_to_read and self.npart_file[ifile,itype] > 0:

                            # May not have opened the snapshot yet
                            if snap is None:
                                snap = self.open_snap_file(ifile)

                            # Read in the data
                            if dataset not in result[itype]:
                                result[itype][dataset] = []
                            result[itype][dataset].append(snap["PartType%d/%s" % (itype, dataset)][first_to_read:last_to_read+1,...])

            # Check if we're done
            if np.all(last_in_snap >= last_in_fof):
                break

            # Advance to next file
            first_in_snap += self.npart_file[ifile,:]

        # Concatenate arrays read from each file
        for itype in range(6):
            for dataset in result[itype].keys():
                result[itype][dataset] = np.concatenate(result[itype][dataset], axis=0)

        return result


