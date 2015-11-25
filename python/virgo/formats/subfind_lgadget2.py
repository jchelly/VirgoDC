#!/bin/env python
#
# Classes to read FoF/Subfind output from L-Gadget2 and L-SubFind
#

import numpy as np
from virgo.util.read_binary import BinaryFile
from virgo.util.peano       import peano_hilbert_key_inverses, peano_hilbert_keys
from virgo.util.exceptions  import SanityCheckFailedException


def cosma_file_path(itype, isnap, ifile):
    """Calculate location of Millennium files on Cosma"""
    idest = (ifile+3*(63-isnap)) % 14
    if itype == 0:
        return "/milli%2.2i/d%2.2i/snapshot/snap_millennium_%3.3i.%i" % (idest,isnap,isnap,ifile)
    elif itype == 1:
        return "/milli%2.2i/d%2.2i/group_tab/group_tab_%3.3i.%i" % (idest,isnap,isnap,ifile)
    elif itype == 2:
        return "/milli%2.2i/d%2.2i/sub_tab/sub_tab_%3.3i.%i" % (idest,isnap,isnap,ifile)
    elif itype == 3:
        return "/milli%2.2i/d%2.2i/sub_ids/sub_ids_%3.3i.%i" % (idest,isnap,isnap,ifile)
    elif itype == 4:
        return "/milli%2.2i/d%2.2i/group_ids/group_ids_%3.3i.%i" % (idest,isnap,isnap,ifile)


class GroupTabFile(BinaryFile):
    """
    Class for reading Millennium-1 group_tab files written by L-Gadget2
    """
    def __init__(self, fname, *args):
        BinaryFile.__init__(self, fname, *args)

        # Header
        self.add_dataset("Ngroups",    np.int32)
        self.add_dataset("Nids",       np.int32)
        self.add_dataset("TotNgroups", np.int32)
        self.add_dataset("NTask",     np.int32)
        
        # Establish endian-ness by sanity check on number of files
        nfiles = self["NTask"][...]
        if nfiles < 1 or nfiles > 65535:
            self.enable_byteswap(True)

        # FoF group info
        ngroups = self["Ngroups"][...]
        self.add_dataset("GroupLen",    np.int32, (ngroups,))
        self.add_dataset("GroupOffset", np.int32, (ngroups,))
        self.add_dataset("GroupMinLen", np.int32)
        minlen = self["GroupMinLen"][...]
        self.add_dataset("Count", np.int32, (minlen,))

    def sanity_check(self):

        ngroups     = self["Ngroups"][...]
        nids        = self["Nids"][...]
        grouplen    = self["GroupLen"][...]
        groupoffset = self["GroupOffset"][...]

        if sum(grouplen) != nids:
            raise SanityCheckFailedException("Sum of group sizes does not equal number of particle IDs")

        if any(groupoffset < 0) or any(groupoffset >= nids):
            raise SanityCheckFailedException("Group offset out of range")

        if ngroups > 1:
            if any(groupoffset[1:] != groupoffset[:-1] + grouplen[:-1]):
                raise SanityCheckFailedException("Group offset not equal to (offset+length) of previous group")

        return None


class GroupIDsFile(BinaryFile):
    """
    Class for reading Millennium-1 group_ids files written by L-Gadget2
    """
    def __init__(self, fname, id_bytes=8, *args):
        BinaryFile.__init__(self, fname, *args)

        # Header
        self.add_dataset("Ngroups",    np.int32)
        self.add_dataset("Nids",       np.int32)
        self.add_dataset("TotNgroups", np.int32)
        self.add_dataset("NTask",      np.int32)
        
        # Establish endian-ness by sanity check on number of files
        nfiles = self["NFiles"][...]
        if nfiles < 1 or nfiles > 65535:
            self.enable_byteswap(True)

        # We also need to know the data types used for particle IDs
        if id_bytes == 4:
            self.id_type = np.uint32
        elif id_bytes == 8:
            self.id_type = np.uint64
        else:
            raise ValueError("id_bytes must be 4 or 8")

        # Read header
        Nids = self["Nids"][...]

        # Particle IDs
        self.add_dataset("GroupIDs",   self.id_type, (Nids,))

    def sanity_check(self):
        
        ids = self["GroupIDs"][...]
        if np.any(ids < 0):
            raise SanityCheckFailedException("Found negative ID")

        # Split ID into particle ID and hash key
        key = np.right_shift(ids, 34)
        ids = ids - np.left_shift(key, 34)

        # Note: hash table size and max ID hard coded here
        if np.any(key < 0) or np.any(key >= 256**3):
            raise SanityCheckFailedException("Hash key out of range")
        if np.any(ids<0) or np.any(ids>2160**3):
            raise SanityCheckFailedException("Particle ID out of range")
        if sum(ids==0) > 1:
            raise SanityCheckFailedException("Found multiple zero IDs")

        # Return None if everything looks ok
        return None


class SubTabFile(BinaryFile):
    """
    Class for reading Millennium-1 sub_tab files written by L-SubFind
    """
    def __init__(self, fname, id_bytes=8, *args):
        BinaryFile.__init__(self, fname, *args)

        # Header
        self.add_dataset("Ngroups",    np.int32)
        self.add_dataset("Nids",       np.int32)
        self.add_dataset("TotNgroups", np.int32)
        self.add_dataset("NFiles",     np.int32)
        self.add_dataset("Nsubhalos",  np.int32)
        
        # Establish endian-ness by sanity check on number of files
        nfiles = self["NFiles"][...]
        if nfiles < 1 or nfiles > 65535:
            self.enable_byteswap(True)

        # We also need to know the data types used for particle IDs
        if id_bytes == 4:
            self.id_type = np.uint32
        elif id_bytes == 8:
            self.id_type = np.uint64
        else:
            raise ValueError("id_bytes must be 4 or 8")

        # Read header
        ngroups    = self["Ngroups"][...]
        nids       = self["Nids"][...]
        nfiles     = self["NFiles"][...]
        nsubgroups = self["Nsubhalos"][...]

        # FoF group info
        self.add_dataset("NsubPerHalo",    np.int32, (ngroups,))
        self.add_dataset("FirstSubOfHalo", np.int32, (ngroups,))

        # Subhalo info
        self.add_dataset("SubLen",        np.int32, (nsubgroups,))
        self.add_dataset("SubOffset",     np.int32, (nsubgroups,))
        self.add_dataset("SubParentHalo", np.int32, (nsubgroups,))

        # Spherical overdensity masses and radii
        self.add_dataset("Halo_M_Mean200",    np.float32, (ngroups,))
        self.add_dataset("Halo_R_Mean200",    np.float32, (ngroups,))
        self.add_dataset("Halo_M_Crit200",    np.float32, (ngroups,))
        self.add_dataset("Halo_R_Crit200",    np.float32, (ngroups,))
        self.add_dataset("Halo_M_TopHat200",  np.float32, (ngroups,))
        self.add_dataset("Halo_R_TopHat200",  np.float32, (ngroups,))
        
        # Subhalo properties
        self.add_dataset("SubPos",         np.float32, (nsubgroups,3))
        self.add_dataset("SubVel",         np.float32, (nsubgroups,3))
        self.add_dataset("SubVelDisp",     np.float32, (nsubgroups,))
        self.add_dataset("SubVmax",        np.float32, (nsubgroups,))
        self.add_dataset("SubSpin",        np.float32, (nsubgroups,3))
        self.add_dataset("SubMostBoundID", self.id_type, (nsubgroups,))
        self.add_dataset("SubHalfMass",    np.float32, (nsubgroups,))

    def sanity_check(self):

        ngroups    = self["Ngroups"][...]
        nids       = self["Nids"][...]
        nfiles     = self["NFiles"][...]
        nsubgroups = self["Nsubhalos"][...]

        nsubperhalo    = self["NsubPerHalo"][...]
        firstsubofhalo = self["FirstSubOfHalo"][...]
        sublen         = self["SubLen"][...]
        suboffset      = self["SubOffset"][...]
        subparenthalo  = self["SubParentHalo"][...]

        # Check assignment of subhalos to halos
        if np.sum(nsubperhalo) != nsubgroups:
            raise SanityCheckFailedException("Sum of NSubPerHalo is wrong")
        ind = nsubperhalo > 0
        if np.any(firstsubofhalo[ind]<0) or np.any(firstsubofhalo[ind]>=nsubgroups):
            raise SanityCheckFailedException("FirstSubOfHalo out of range")
        if np.any(suboffset<0) or np.any(suboffset+sublen>nids):
            raise SanityCheckFailedException("Subhalo particle index(es) out of range")

        # Check subhalo parent group index
        parent = np.repeat(np.arange(ngroups, dtype=np.int32), nsubperhalo)
        if np.any(subparenthalo != parent):
            raise SanityCheckFailedException("Subhalo parent halo index is wrong")

        # Check quantities which should be finite
        for name in ("Halo_M_Mean200",   "Halo_R_Mean200", 
                     "Halo_M_Crit200",   "Halo_R_Crit200", 
                     "Halo_M_TopHat200", "Halo_R_TopHat200",
                     "SubPos", "SubVel", "SubVelDisp",
                     "SubVmax", "SubSpin", "SubHalfMass"):
            data = self[name][...]
            if np.any(np.logical_not(np.isfinite(data))):
                raise SanityCheckFailedException("Quantity %s has non-finite value" % name)
        
        # Return None if everything looks ok
        return None


class SubIDsFile(BinaryFile):
    """
    Class for reading Millennium-1 sub_ids files written by L-SubFind
    """
    def __init__(self, fname, id_bytes=8, *args):
        BinaryFile.__init__(self, fname, *args)

        # Header
        self.add_dataset("Ngroups",    np.int32)
        self.add_dataset("Nids",       np.int32)
        self.add_dataset("TotNgroups", np.int32)
        self.add_dataset("NFiles",     np.int32)
        
        # Establish endian-ness by sanity check on number of files
        nfiles = self["NFiles"][...]
        if nfiles < 1 or nfiles > 65535:
            self.enable_byteswap(True)

        # We also need to know the data types used for particle IDs
        if id_bytes == 4:
            self.id_type = np.uint32
        elif id_bytes == 8:
            self.id_type = np.uint64
        else:
            raise ValueError("id_bytes must be 4 or 8")

        # Read header
        Nids = self["Nids"][...]

        # Particle IDs
        self.add_dataset("GroupIDs",   self.id_type, (Nids,))

    def sanity_check(self):
        
        ids = self["GroupIDs"][...]
        if np.any(ids < 0):
            raise SanityCheckFailedException("Found negative ID")

        # Split ID into particle ID and hash key
        key = np.right_shift(ids, 34)
        ids = ids - np.left_shift(key, 34)

        # Note: hash table size and max ID hard coded here
        if np.any(key < 0) or np.any(key >= 256**3):
            raise SanityCheckFailedException("Hash key out of range")
        if np.any(ids<0) or np.any(ids>2160**3):
            raise SanityCheckFailedException("Particle ID out of range")
        if sum(ids==0) > 1:
            raise SanityCheckFailedException("Found multiple zero IDs")

        # Return None if everything looks ok
        return None
        

class SnapshotFile(BinaryFile):
    """
    Class for reading Millennium-1 snapshot files
    """
    def __init__(self, fname, *args):
        BinaryFile.__init__(self, fname, *args)

        # Header
        self.start_fortran_record(auto_byteswap=True)
        self.add_attribute("Header/NumPart_ThisFile",  np.uint32,  (6,))
        self.add_attribute("Header/MassTable",         np.float64, (6,))
        self.add_attribute("Header/Time",              np.float64)
        self.add_attribute("Header/Redshift",          np.float64)
        self.add_attribute("Header/Flag_Sfr",          np.int32)
        self.add_attribute("Header/Flag_Feedback",     np.int32)
        self.add_attribute("Header/NumPart_Total",     np.uint32,  (6,))
        self.add_attribute("Header/Flag_Cooling",      np.int32)
        self.add_attribute("Header/NumFilesPerSnapshot", np.int32)
        self.add_attribute("Header/BoxSize",         np.float64)
        self.add_attribute("Header/Omega0",          np.float64)
        self.add_attribute("Header/OmegaLambda",     np.float64)
        self.add_attribute("Header/HubbleParam",     np.float64)
        self.add_attribute("Header/Flag_StellarAge", np.int32)
        self.add_attribute("Header/Flag_Metals",     np.int32)
        self.add_attribute("Header/HashTabSize",     np.int32)
        self.add_attribute("Header/fill",            np.uint8, (84,))
        self.end_fortran_record()

        # Get number of particles in this file
        n = self["Header"].attrs["NumPart_ThisFile"][1]

        # Coordinates
        self.start_fortran_record()
        self.add_dataset("PartType1/Coordinates", np.float32, (n,3))
        self.end_fortran_record()

        # Velocities
        self.start_fortran_record()
        self.add_dataset("PartType1/Velocities", np.float32, (n,3))
        self.end_fortran_record()

         # IDs
        self.start_fortran_record()
        self.add_dataset("PartType1/ParticleIDs", np.uint64, (n,))
        self.end_fortran_record()

        # Range of hash cells in this file
        self.start_fortran_record()
        self.add_dataset("first_hash_cell", np.int32)
        self.add_dataset("last_hash_cell",  np.int32)
        self.end_fortran_record()
        
        # Calculate how many hash cells we have in this file
        nhash = self["last_hash_cell"][...] - self["first_hash_cell"][...] + 1

        # Location of first particle in each cell relative to start of file
        self.start_fortran_record()
        self.add_dataset("blockid", np.int32, (nhash,))
        self.end_fortran_record()

    def sanity_check(self):

        n       = self["Header"]
        boxsize = self["Header"].attrs["BoxSize"]
        nptot   = self["Header"].attrs["NumPart_Total"]
        hashtabsize = self["Header"].attrs["HashTabSize"]
        nptot = nptot[1] + (nptot[2] << 32)

        # Determine hashbits
        hashbits = 1
        while hashtabsize > 2:
            hashtabsize /= 2
            hashbits += 1
        hashbits /= 3 # bits per dimension
        del hashtabsize

        # Check positions
        pos = self["PartType1/Coordinates"][...]
        if not(np.all(np.isfinite(pos))):
            raise SanityCheckFailedException("Particle coordinate is not finite")
        if np.any(pos < 0.0) or np.any(pos > boxsize):
            raise SanityCheckFailedException("Particle coordinate out of range")
        # (don't dealloc positions yet - needed for hash table check)

        # Check velocities
        vel = self["PartType1/Velocities"][...]
        if not(np.all(np.isfinite(vel))):
            raise SanityCheckFailedException("Particle velocity is not finite")
        if np.any(abs(vel) > 1.0e6):
            raise SanityCheckFailedException("Suspiciously high velocity")
        del vel

        # Check IDs
        ids = self["PartType1/ParticleIDs"][...]
        if np.any(ids < 0) or np.any(ids > nptot):
            raise SanityCheckFailedException("Particle ID out of range")
        del ids

        # Read hash table
        first_hash_cell = self["first_hash_cell"][...]
        last_hash_cell  = self["last_hash_cell"][...]
        blockid         = self["blockid"][...]

        # Calculate hash key for each particle
        key = np.zeros(pos.shape[0], dtype=np.int32) - 1
        for i in range(last_hash_cell-first_hash_cell):
            key[blockid[i]:blockid[i+1]] = i + first_hash_cell
        key[blockid[-1]:] = last_hash_cell

        # Convert hash keys to grid coordinates
        ix, iy, iz = peano_hilbert_key_inverses(key, hashbits)

        # Convert grid coordinates to physical coords of cell centre
        cellsize = boxsize / (2**hashbits)
        cell_pos = np.empty_like(pos)
        cell_pos[:,0] = ix * cellsize + 0.5*cellsize
        cell_pos[:,1] = iy * cellsize + 0.5*cellsize
        cell_pos[:,2] = iz * cellsize + 0.5*cellsize

        # Ensure all particles are within cells
        for i in range(3):
            dx = np.abs(pos[:,i]-cell_pos[:,i])
            ind = dx>0.5*boxsize
            dx[ind] = boxsize - dx[ind]
            if any(dx > 1.001*0.5*cellsize):
                raise SanityCheckFailedException("Particle not in correct hash cell")
        
        # Return None if everything looks ok
        return None


class Snapshot():
    """
    Class for reading parts of a Millennium snapshot
    using the hash table.
    """
    def __init__(self, basedir, basename, isnap):
        """
        Open a new snapshot

        basedir:  name of directory with snapdir_??? and postproc_??? subdirs
        basename: snapshot file name before the last underscore (e.g. snap_millennium)
        isnap:    which snapshot to read
        """
        
        # Store file name info
        self.basedir    = basedir
        self.basename   = basename
        self.isnap      = isnap

        # Get box size, size of hash grid etc from first file
        snap = self.open_file(self.isnap, 0)
        self.nhash   = snap["Header"].attrs["HashTabSize"]
        self.boxsize = snap["Header"].attrs["BoxSize"]
        self.nfiles  = snap["Header"].attrs["NumFilesPerSnapshot"]
        self.ncell = 1
        self.bits  = 0
        while self.ncell**3 < self.nhash:
            self.ncell *= 2
            self.bits += 1

        # Initialise cache for hash table
        self.hash_table = {}

    def open_file(self, isnap, ifile):
        """
        Open the specified snapshot file
        """
        if self.basedir != "COSMA":
            fname = "%s/snapdir_%03d/%s_%03d.%d" % (self.basedir, isnap, 
                                                    self.basename, isnap, ifile)
        else:
            fname = "/gpfs/data/Millennium/"+cosma_file_path(0, isnap, ifile)
        return SnapshotFile(fname) 
    
    def read_region(self, coords):
        """
        Use the hash table to extract a region from a Millennium snapshot
        
        coords: region to read in the form (xmin, xmax, ymin, ymax, zmin, zmax)
        """

        # Identify requested hash cells
        cellsize = self.boxsize / self.ncell
        icoords = np.floor(np.asarray(coords, dtype=float) / cellsize)
        idx = np.indices((icoords[1]-icoords[0]+1,
                          icoords[3]-icoords[2]+1,
                          icoords[5]-icoords[4]+1))
        ix = np.mod(idx[0].flatten() + icoords[0], self.ncell)
        iy = np.mod(idx[1].flatten() + icoords[2], self.ncell)
        iz = np.mod(idx[2].flatten() + icoords[4], self.ncell)
        keys = peano_hilbert_keys(ix, iy, iz, self.bits)

        # Read particles in these cells
        return self.read_cells(keys)

    def read_cells(self, keys):
        """
        Use the hash table to read particles with specified hash keys
        """

        # Flag requested grid cells
        hashgrid = np.zeros(self.nhash, dtype=np.bool)
        hashgrid[keys] = True

        # Loop over files to read
        pos = []
        vel = []
        ids = []
        for ifile in range(self.nfiles):

            snap = None

            # Check if we have cached hash table data for this file
            # so we can avoid opening it if it contains no selected cells
            if ifile in self.hash_table:
                first_hash_cell, last_hash_cell = self.hash_table[ifile]
            else:
                snap = self.open_file(self.isnap, ifile)
                first_hash_cell = snap["first_hash_cell"][...]
                last_hash_cell  = snap["last_hash_cell"][...]
                self.hash_table[ifile] = (first_hash_cell, last_hash_cell)

            # Check if we have some cells to read from this file
            read_cell = hashgrid[first_hash_cell:last_hash_cell+1]
            if np.any(read_cell):

                # Need to read this file
                if snap is None:
                    snap = self.open_file(self.isnap, ifile)
                    
                # Read number of particles in this file
                npart = snap["Header"].attrs["NumPart_ThisFile"][1]

                # Read this part of the hash table
                first_in_this_cell = snap["blockid"][...]
                first_in_next_cell = np.empty_like(first_in_this_cell)
                first_in_next_cell[:-1] = first_in_this_cell[1:]
                first_in_next_cell[-1] = npart
                num_in_cell = first_in_next_cell - first_in_this_cell

                # Determine which particles we need to read
                read_particle = np.repeat(read_cell, num_in_cell)

                # Read the selected particles from this file
                pos.append(snap["PartType1/Coordinates"][read_particle,:])
                vel.append(snap["PartType1/Velocities"][read_particle,:])
                ids.append(snap["PartType1/ParticleIDs"][read_particle])

        # Assemble output arrays
        pos = np.concatenate(pos, axis=0)
        vel = np.concatenate(vel, axis=0)
        ids = np.concatenate(ids)

        return pos, vel, ids
