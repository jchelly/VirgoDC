#!/bin/env python
#
# Classes to read SubFind output from P-Gadget3
# (at least the versions used for Millennium-2, MR7, Aquarius and Phoenix)
#

import os
import numpy as np
from collections import Mapping
from virgo.util.read_binary import BinaryFile
from virgo.util.exceptions  import SanityCheckFailedException
from virgo.util.read_multi import read_multi
from virgo.formats import gadget_snapshot

class GroupIDsFile(BinaryFile):
    """
    Class for reading group_ids files written by P-Gadget3
    """
    def __init__(self, fname, 
                 id_bytes=8, float_bytes=4,
                 *args):
        BinaryFile.__init__(self, fname, *args)
        
        # Get number of bytes used to store IDs
        if id_bytes == 4:
            self.id_type = np.uint32
        elif id_bytes == 8:
            self.id_type = np.uint64
        else:
            raise ValueError("id_bytes must be 4 or 8")

        self.add_dataset("Ngroups",    np.int32)
        self.add_dataset("TotNgroups", np.int32)
        self.add_dataset("Nids",       np.int32)
        self.add_dataset("TotNids",    np.int64)
        self.add_dataset("NTask",      np.int32)
        self.add_dataset("SendOffset", np.int32)
        
        # Establish endian-ness by sanity check on number of files
        nfiles = self["NTask"][...]
        if nfiles < 1 or nfiles > 65535:
            self.enable_byteswap(True)
            
        # Read header
        Nids = self["Nids"][...]

        # Add dataset with particle IDs
        self.add_dataset("GroupIDs",   self.id_type, (Nids,))

    def sanity_check(self):

        # Check file has the expected size (e.g. in case ID type is wrong)
        if not(self.all_bytes_used()):
            raise SanityCheckFailedException("File size is incorrect!")

        # Check that IDs don't contain duplicates
        ids = self["GroupIDs"][...]
        idx, counts = np.unique(ids, return_counts=True)
        if np.any(counts != 1):
            raise SanityCheckFailedException("Found duplicate IDs!")

class GroupTabFile(BinaryFile):
    """
    Class for reading group_tab files written by P-Gadget3
    """
    def __init__(self, fname, 
                 id_bytes=8, float_bytes=4,
                 *args):
        BinaryFile.__init__(self, fname, *args)

        # We need to know the data types used for particle IDs
        # and floating point subhalo properties (again, this can't be read from the file).
        if id_bytes == 4:
            self.id_type = np.uint32
        elif id_bytes == 8:
            self.id_type = np.uint64
        else:
            raise ValueError("id_bytes must be 4 or 8")
        if float_bytes == 4:
            self.float_type = np.float32
        elif float_bytes == 8:
            self.float_type = np.float64
        else:
            raise ValueError("float_bytes must be 4 or 8")

        # Define data blocks in the group_tab file
        # Header
        self.add_dataset("Ngroups",    np.int32)
        self.add_dataset("TotNgroups", np.int32)
        self.add_dataset("Nids",       np.int32)
        self.add_dataset("TotNids",    np.int64)
        self.add_dataset("NTask",     np.int32)

        # Establish endian-ness by sanity check on number of files
        nfiles = self["NTask"][...]
        if nfiles < 1 or nfiles > 65535:
            self.enable_byteswap(True)
 
        # FoF group information
        ngroups = self["Ngroups"][...]
        self.add_dataset("GroupLen",          np.int32,        (ngroups,))
        self.add_dataset("GroupOffset",       np.uint32,        (ngroups,)) # Assume uint32 to avoid overflow in AqA1
        self.add_dataset("GroupMass",         self.float_type, (ngroups,))
        self.add_dataset("GroupCofM",         self.float_type, (ngroups,3))
        self.add_dataset("GroupVel",          self.float_type, (ngroups,3))
        self.add_dataset("GroupLenType",      np.int32,        (ngroups,6))
        self.add_dataset("GroupMassType",     self.float_type, (ngroups,6))

    def sanity_check(self):

        # Check file has the expected size (e.g. in case ID type is wrong)
        if not(self.all_bytes_used()):
            raise SanityCheckFailedException("File size is incorrect!")

        # Check sum over particle types equals total number of particles in each group
        grouplen     = self["GroupLen"][...]
        grouplentype = self["GroupLenType"][...]
        if np.any(np.sum(grouplentype, axis=1, dtype=np.int64) != grouplen):
            raise SanityCheckFailedException("GroupLen not consistent with GroupLenType!")

        # Each offset should be previous offset plus length
        if self["Nids"][...] > 1:
            groupoffset = self["GroupOffset"][...]        
            if np.any(groupoffset[1:] != groupoffset[:-1] + grouplen[:-1]):
                raise SanityCheckFailedException("Offsets and lengths not consistent!")
        
        # Check sum of mass over particle types equals total mass of particles in each group
        groupmass     = self["GroupMass"][...]
        groupmasstype = self["GroupMassType"][...]
        mass_sum = np.sum(groupmasstype, axis=1, dtype=np.float64)
        if np.any(np.abs(mass_sum-groupmass)/groupmass > 1.0e-5):
            raise SanityCheckFailedException("GroupMass not consistent with GroupMassType!")
        


class SubTabFile(BinaryFile):
    """
    Class for reading sub_tab files written by P-Gadget3
    """
    def __init__(self, fname, 
                 SO_VEL_DISPERSIONS=False,
                 SO_BAR_INFO=False,
                 WRITE_SUB_IN_SNAP_FORMAT=False,
                 id_bytes=8, float_bytes=4,
                 *args):
        BinaryFile.__init__(self, fname, *args)

        # Haven't implemented these
        if WRITE_SUB_IN_SNAP_FORMAT:
            raise NotImplementedError("Subfind outputs in type 2 binary snapshot format are not implemented")
        if SO_BAR_INFO:
            raise NotImplementedError("Subfind outputs with SO_BAR_INFO set are not implemented")

        # These parameters, which correspond to macros in Gadget's Config.sh,
        # modify the file format. The file cannot be read correctly unless these
        # are known - their values are not stored in the output.
        self.WRITE_SUB_IN_SNAP_FORMAT = WRITE_SUB_IN_SNAP_FORMAT
        self.SO_VEL_DISPERSIONS       = SO_VEL_DISPERSIONS
        self.SO_BAR_INFO              = SO_BAR_INFO

        # We also need to know the data types used for particle IDs
        # and floating point subhalo properties (again, this can't be read from the file).
        if id_bytes == 4:
            self.id_type = np.uint32
        elif id_bytes == 8:
            self.id_type = np.uint64
        else:
            raise ValueError("id_bytes must be 4 or 8")
        if float_bytes == 4:
            self.float_type = np.float32
        elif float_bytes == 8:
            self.float_type = np.float64
        else:
            raise ValueError("float_bytes must be 4 or 8")
       
        # Define data blocks in the subhalo_tab file
        # Header
        self.add_dataset("Ngroups",    np.int32)
        self.add_dataset("TotNgroups", np.int32)
        self.add_dataset("Nids",       np.int32)
        self.add_dataset("TotNids",    np.int64)
        self.add_dataset("NTask",     np.int32)
        self.add_dataset("Nsubgroups",    np.int32)
        self.add_dataset("TotNsubgroups", np.int32)

        # Establish endian-ness by sanity check on number of files
        nfiles = self["NTask"][...]
        if nfiles < 1 or nfiles > 65535:
            self.enable_byteswap(True)
            
        # FoF group information
        ngroups = self["Ngroups"][...]
        self.add_dataset("GroupLen",          np.int32,        (ngroups,))
        self.add_dataset("GroupOffset",       np.uint32,        (ngroups,)) # Assume uint32 to avoid overflow in AqA1
        self.add_dataset("GroupMass",         self.float_type, (ngroups,))
        self.add_dataset("GroupPos",          self.float_type, (ngroups,3))
        self.add_dataset("Halo_M_Mean200",    self.float_type, (ngroups,))
        self.add_dataset("Halo_R_Mean200",    self.float_type, (ngroups,))
        self.add_dataset("Halo_M_Crit200",    self.float_type, (ngroups,))
        self.add_dataset("Halo_R_Crit200",    self.float_type, (ngroups,))
        self.add_dataset("Halo_M_TopHat200",  self.float_type, (ngroups,))
        self.add_dataset("Halo_R_TopHat200",  self.float_type, (ngroups,))
        
        # Optional extra FoF fields
        if SO_VEL_DISPERSIONS:
            self.add_dataset("VelDisp_Mean200",    self.float_type, (ngroups,))
            self.add_dataset("VelDisp_Crit200",    self.float_type, (ngroups,))
            self.add_dataset("VelDisp_TopHat200",  self.float_type, (ngroups,))

        # FoF contamination info
        self.add_dataset("ContaminationLen",       np.int32,        (ngroups,))
        self.add_dataset("ContaminationMass",      self.float_type, (ngroups,))
        
        # Count and offset to subhalos in each FoF group
        self.add_dataset("Nsubs",                  np.int32,        (ngroups,))
        self.add_dataset("FirstSub",               np.int32,        (ngroups,))
        
        # Subhalo properties
        nsubgroups = self["Nsubgroups"][...]
        self.add_dataset("SubLen",     np.int32, (nsubgroups,))
        self.add_dataset("SubOffset",  np.int32, (nsubgroups,))
        self.add_dataset("SubParent",  np.int32, (nsubgroups,))
        self.add_dataset("SubMass",    self.float_type, (nsubgroups,))
        self.add_dataset("SubPos",     self.float_type, (nsubgroups,3))
        self.add_dataset("SubVel",     self.float_type, (nsubgroups,3))
        self.add_dataset("SubCofM",    self.float_type, (nsubgroups,3))
        self.add_dataset("SubSpin",    self.float_type, (nsubgroups,3))
        self.add_dataset("SubVelDisp",     self.float_type, (nsubgroups,))
        self.add_dataset("SubVmax",        self.float_type, (nsubgroups,))
        self.add_dataset("SubRVmax",       self.float_type, (nsubgroups,))
        self.add_dataset("SubHalfMass",    self.float_type, (nsubgroups,))
        self.add_dataset("SubMostBoundID", self.id_type,    (nsubgroups,))
        self.add_dataset("SubGrNr",        np.int32,        (nsubgroups,))

    def sanity_check(self):

        # Check file has the expected size (e.g. in case ID type is wrong)
        if not(self.all_bytes_used()):
            raise SanityCheckFailedException("File size is incorrect!")
        
        totnids       = self["TotNids"][...]
        totngroups    = self["TotNgroups"][...]
        totnsubgroups = self["TotNsubgroups"][...]
        
        # Checks on header
        if totnids < 0 or totngroups < 0 or totnsubgroups < 0:
            raise SanityCheckFailedException("Negative number of groups/subgroups/IDs")

        # Checks on group properties
        if np.any(self["GroupLen"][...] < 0):
            raise SanityCheckFailedException("Found group with non-positive length!")
        if totnids < 2**31:
            # Offsets overflow if we have too many particles (e.g. Millennium-2), 
            # so this test is expected to fail in that case
            if np.any(self["GroupOffset"][...] < 0) or np.any(self["GroupOffset"][...]+self["GroupLen"][...] > totnids):
                raise SanityCheckFailedException("Found group with offset out of range!")

        # Check pointers from groups to subgroups are sane
        firstsub = self["FirstSub"][...]
        nsubs    = self["Nsubs"][...]
        if np.any(nsubs < 0):
            raise SanityCheckFailedException("Negative number of subgroups in group!")
        ind = nsubs > 0
        if np.any(firstsub[ind] < 0) or np.any(firstsub[ind]+nsubs[ind] > totnsubgroups):
            raise SanityCheckFailedException("Found group with subgroup index out of range!")

        # Check some group properties that we expect to be finite and non-negative
        for prop in ("GroupMass",
                     "Halo_M_Mean200",  "Halo_R_Mean200",
                     "Halo_M_Crit200",  "Halo_R_Crit200",
                     "Halo_M_TopHat200","Halo_R_TopHat200"):
            data = self[prop][...]
            if not(np.all(np.isfinite(data))):
                raise Exception("Found non-finite value in dataset %s" % prop)
            if not(np.all(data >= 0.0)):
                raise Exception("Found negative value in dataset %s" % prop)

        # Checks on subgroup properties
        if np.any(self["SubLen"][...] < 0):
            raise SanityCheckFailedException("Found subgroup with non-positive length!")
        if totnids < 2**31:
            # Offsets overflow if we have too many particles (e.g. Millennium-2),
            # so this test is expected to fail in that case
            if np.any(self["SubOffset"][...] < 0) or np.any(self["SubOffset"][...]+self["SubLen"][...] > totnids):
                raise SanityCheckFailedException("Found group with offset out of range!")

        # Check some subgroup properties that we expect to be finite and (in some cases) non-negative
        for prop in ("SubMass","SubPos","SubVel","SubCofM","SubSpin","SubVelDisp",
                     "SubVmax","SubRVmax","SubHalfMass"):
            data = self[prop][...]
            if not(np.all(np.isfinite(data))):
                raise Exception("Found non-finite value in dataset %s" % prop)
            if not(np.all(data >= 0.0)) and prop in ("SubMass","SubVelDisp","SubVmax","SubRVmax","SubHalfMass"):
                raise Exception("Found negative value in dataset %s" % prop)

        # Check SubGrNr is in range and in the expected order
        subgrnr = self["SubGrNr"][...]
        if np.any(subgrnr<0) or np.any(subgrnr>=totngroups):
            raise SanityCheckFailedException("Subgroup's SubGrNr out of range!")
        if subgrnr.shape[0] > 1:
            if np.any(subgrnr[1:] < subgrnr[:-1]):
                raise SanityCheckFailedException("Subgroup SubGrNr's are not in ascending order!")



class SubIDsFile(BinaryFile):
    """
    Class for reading sub_ids files written by P-Gadget3
    """
    def __init__(self, fname, id_bytes=8, *args):
        BinaryFile.__init__(self, fname, *args)

        # We need to know the data type used for particle IDs
        if id_bytes == 4:
            self.id_type = np.uint32
        elif id_bytes == 8:
            self.id_type = np.uint64
        
        # Define data blocks in the subhalo_tab file
        # Header
        self.add_dataset("Ngroups",    np.int32)
        self.add_dataset("TotNgroups", np.int32)
        self.add_dataset("Nids",       np.int32)
        self.add_dataset("TotNids",    np.int64)
        self.add_dataset("NTask",      np.int32)
        self.add_dataset("SendOffset", np.int32)

        # Establish endian-ness by sanity check on number of files
        nfiles = self["NTask"][...]
        if nfiles < 1 or nfiles > 65535:
            self.enable_byteswap(True)
            
        # Read header
        Nids = self["Nids"][...]

        # Add dataset with particle IDs
        self.add_dataset("GroupIDs",   self.id_type, (Nids,))

    def sanity_check(self):

        # Check file has the expected size (e.g. in case ID type is wrong)
        if not(self.all_bytes_used()):
            raise SanityCheckFailedException("File size is incorrect!")

        # Check that IDs don't contain duplicates
        ids = self["GroupIDs"][...]
        idx, counts = np.unique(ids, return_counts=True)
        if np.any(counts != 1):
            raise SanityCheckFailedException("Found duplicate IDs!")



class GroupCatalogue(Mapping):
    """
    Class for reading the complete group catalogue for
    a snapshot into memory.

    This class acts as a dictionary where the keys are dataset
    names and the values are numpy arrays with the data.
    """
    def __init__(self, basedir, isnap, datasets=None,
                 SO_VEL_DISPERSIONS=False,
                 SO_BAR_INFO=False,
                 WRITE_SUB_IN_SNAP_FORMAT=False,
                 id_bytes=8, float_bytes=4):


        # Default datasets to read
        if datasets is None:
            datasets =  ["GroupLen",  "GroupOffset",  "GroupMass",  "GroupPos", 
                         "Halo_M_Mean200",  "Halo_R_Mean200",  "Halo_M_Crit200",  
                         "Halo_R_Crit200",  "Halo_M_TopHat200",  "Halo_R_TopHat200", 
                         "ContaminationLen",  "ContaminationMass",  "Nsubs",  
                         "FirstSub",  "SubLen",  "SubOffset",  "SubParent",  
                         "SubMass",  "SubPos",  "SubVel",  "SubCofM",  "SubSpin",
                         "SubVelDisp",  "SubVmax",  "SubRVmax",  "SubHalfMass",  
                         "SubMostBoundID", "SubGrNr"]
            if SO_VEL_DISPERSIONS:
                datasets += ["VelDisp_Mean200",  "VelDisp_Crit200",  "VelDisp_TopHat200"]

        # Construct format string for file names
        fname_fmt = ("%s/groups_%03d/subhalo_tab_%03d" % (basedir, isnap, isnap)) + ".%(i)d"

        # Get number of files
        f = SubTabFile(fname_fmt % {"i":0}, 
                       id_bytes=id_bytes, float_bytes=float_bytes,
                       SO_VEL_DISPERSIONS=SO_VEL_DISPERSIONS, 
                       SO_BAR_INFO=SO_BAR_INFO,
                       WRITE_SUB_IN_SNAP_FORMAT=WRITE_SUB_IN_SNAP_FORMAT)
        nfiles = f["NTask"][...]
        del f
        
        # Read the catalogue data
        self._items = read_multi(fname_fmt, np.arange(nfiles), datasets, 
                                 file_type=SubTabFile,
                                 id_bytes=id_bytes, float_bytes=float_bytes,
                                 SO_VEL_DISPERSIONS=SO_VEL_DISPERSIONS, 
                                 SO_BAR_INFO=SO_BAR_INFO,
                                 WRITE_SUB_IN_SNAP_FORMAT=WRITE_SUB_IN_SNAP_FORMAT)

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
    
    Note: P-Gadget3 outputs only contain enough information to
    do this if there is only one particle type in the groups, so
    here we assume that there are only type-1 particles in the FoF
    and subfind groups (e.g. as in Millennium-2 or Aq-A-1).
    """
    def __init__(self, basedir, basename, isnap,
                 SO_VEL_DISPERSIONS=False,
                 SO_BAR_INFO=False,
                 WRITE_SUB_IN_SNAP_FORMAT=False,
                 id_bytes=8, float_bytes=4):
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

        # Store file format info
        self.SO_VEL_DISPERSIONS = SO_VEL_DISPERSIONS
        self.SO_BAR_INFO        = SO_BAR_INFO
        self.WRITE_SUB_IN_SNAP_FORMAT = WRITE_SUB_IN_SNAP_FORMAT
        self.id_bytes           = id_bytes
        self.float_bytes        = float_bytes

        # Read group lengths from all files
        self.foflen = []
        self.sublen = []
        self.firstsub = []
        self.nsubs  = []
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
                self.foflen.append(sub["GroupLen"][...])
                self.nsubs.append(sub["Nsubs"][...])
                self.firstsub.append(sub["FirstSub"][...])
            nsub = sub["Nsubgroups"][...]
            if nsub > 0:
                self.sublen.append(sub["SubLen"][...])

            # Store number of groups in each file
            if ifile == 0:
                self.nfof_file = -np.ones(nfiles, dtype=np.int32)
                self.nsub_file = -np.ones(nfiles, dtype=np.int32)
            self.nfof_file[ifile] = nfof
            self.nsub_file[ifile] = nsub

            ifile += 1
            
        self.foflen   = np.concatenate(self.foflen,   axis=0)
        self.nsubs    = np.concatenate(self.nsubs,    axis=0)
        self.firstsub = np.concatenate(self.firstsub, axis=0)
        self.sublen   = np.concatenate(self.sublen,   axis=0)

        # Calculate offset to each fof group
        self.fof_offset = np.cumsum(self.foflen, dtype=np.int64, axis=0) - self.foflen

        # Find number of snapshot files in this snapshot
        snap = self.open_snap_file(0)
        self.num_snap_files = snap["Header"].attrs["NumFilesPerSnapshot"]
        self.npart_file = -np.ones((self.num_snap_files,), dtype=np.int64)

        # Find total particle number
        nptot = (snap["Header"].attrs["NumPart_Total"].astype(np.int64) + 
                 (snap["Header"].attrs["NumPart_Total_HighWord"].astype(np.int64) << 32))

        # Calculate index of first fof group in each file
        self.first_fof_in_file = np.cumsum(self.nfof_file) - self.nfof_file

    def open_snap_file(self, ifile):
        """
        Open the specified snapshot file
        """
        fname = "%s/snapdir_%03d/%s_%03d.%d" % (self.basedir, self.isnap, 
                                                self.basename, self.isnap, ifile)
        return gadget_snapshot.open(fname) 
    
    def open_subtab_file(self, ifile):
        """
        Open the specified subhalo tab file
        """
        fname = "%s/groups_%03d/subhalo_tab_%03d.%d" % (self.basedir, self.isnap, self.isnap, ifile)

        return SubTabFile(fname, id_bytes=self.id_bytes, float_bytes=self.float_bytes,
                          SO_VEL_DISPERSIONS=self.SO_VEL_DISPERSIONS, 
                          SO_BAR_INFO=self.SO_BAR_INFO,
                          WRITE_SUB_IN_SNAP_FORMAT=self.WRITE_SUB_IN_SNAP_FORMAT)

    def read_fof_group(self, grnr):
        """
        Read pos, vel, id for particles in the specified FoF group

        grnr can either be a single integer giving the position of the group
        in the full catalogue, or a two element sequence with the file number
        and position relative to the start of that file

        Returns a dictionary containing the Coordinates, Velocities
        and ParticleIDs arrays.
        """

        try:
            ifof = int(grnr)
        except TypeError:
            filenum, fofnum = grnr
            ifof = self.first_fof_in_file[filenum] + fofnum

        # Output will be a dictionary with Coordinates, ParticleIDs etc.
        result = {}

        # Determine which particles we need to read
        first_in_fof = self.fof_offset[ifof]
        last_in_fof  = first_in_fof + self.foflen[ifof] - 1
        
        # Now loop over snapshot files and read those we need
        first_in_snap = 0
        for ifile in range(self.num_snap_files):
            
            snap = None

            # Read number of particles in file if necessary
            if self.npart_file[ifile] == -1:
                snap = self.open_snap_file(ifile)
                self.npart_file[ifile] = snap["Header"].attrs["NumPart_ThisFile"][1]

            # Check if any particles we need are in this file
            last_in_snap = first_in_snap + self.npart_file[ifile]

            # Loop over quantities to read
            for dataset in ("Coordinates","Velocities","ParticleIDs"):
                    
                # Check if snapshot has particles of this type
                if last_in_snap >= first_in_snap:

                    # Find range of particles to read from this file
                    first_to_read = first_in_fof - first_in_snap
                    last_to_read  = last_in_fof  - first_in_snap
                    if first_to_read < 0:
                        first_to_read = 0
                    if last_to_read >= self.npart_file[ifile]:
                        last_to_read = self.npart_file[ifile] - 1

                    # Read the particles, if there are any in this file
                    if last_to_read >= first_to_read and self.npart_file[ifile] > 0:

                        # May not have opened the snapshot yet
                        if snap is None:
                            snap = self.open_snap_file(ifile)

                        # Read in the data
                        if dataset not in result:
                            result[dataset] = []
                        result[dataset].append(snap["PartType%d/%s" % (1, dataset)][first_to_read:last_to_read+1,...])

            # Check if we're done
            if last_in_snap >= last_in_fof:
                break

            # Advance to next file
            first_in_snap += self.npart_file[ifile]

        # Concatenate arrays read from each file
        for dataset in result.keys():
            result[dataset] = np.concatenate(result[dataset], axis=0)

        # Now calculate which subfind group each particle belongs to
        if "ParticleIDs" in list(result.keys()):

            # Initialise array to -1 to indicate not in a subgroup
            n = result["ParticleIDs"].shape[0]
            result["SubNr"] = -np.ones(n, dtype=np.int32)

            # Check if we have any subgroups in this group
            if self.nsubs[ifof] > 0:

                # Find range of subgroups in this group
                first_sub = self.firstsub[ifof]
                last_sub  = first_sub + self.nsubs[ifof] - 1

                # Make array with subgroup number for each particle in these subgroups
                sublen = self.sublen[first_sub:last_sub+1]
                subnr  = np.repeat(np.arange(sublen.shape[0], dtype=np.int32), sublen)
                subnr += first_sub

                # Update the part of the output array corresponding to these particles
                result["SubNr"][0:subnr.shape[0]] = subnr

        return result
