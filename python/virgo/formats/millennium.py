#!/bin/env python
#
# Classes to read Millennium-1 output
#

import numpy as np
from ..util.read_binary import BinaryFile
from ..util.peano import peano_hilbert_key_inverses


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

        # Read header
        ngroups    = self["Ngroups"][...]
        nids       = self["Nids"][...]
        nfiles     = self["NTask"][...]

        # FoF group info
        self.add_dataset("GroupLen",    np.int32, (ngroups,))
        self.add_dataset("GroupOffset", np.int32, (ngroups,))

    def sanity_check(self):

        ngroups     = self["Ngroups"][...]
        nids        = self["Nids"][...]
        grouplen    = self["GroupLen"][...]
        groupoffset = self["GroupOffset"][...]

        if sum(grouplen) != nids:
            return "Sum of group sizes does not equal number of particle IDs"

        if any(groupoffset < 0) or any(groupoffset >= nids):
            return "Group offset out of range"

        if ngroups > 1:
            if any(groupoffset[1:] != groupoffset[:-1] + grouplen[:-1]):
                return "Group offset not equal to (offset+length) of previous group"

        return None


class GroupIDsFile(BinaryFile):
    """
    Class for reading Millennium-1 group_ids files written by L-Gadget2
    """
    def __init__(self, fname, *args):
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

        # Read header
        Nids = self["Nids"][...]

        # Particle IDs
        self.add_dataset("GroupIDs",   np.int64, (Nids,))

    def sanity_check(self):
        
        ids = self["GroupIDs"][...]
        if np.any(ids < 0):
            return "Found negative ID"

        # Split ID into particle ID and hash key
        key = np.right_shift(ids, 34)
        ids = ids - np.left_shift(key, 34)

        # Note: hash table size and max ID hard coded here
        if np.any(key < 0) or np.any(key >= 256**3):
            return "Hash key out of range"
        if np.any(ids<0) or np.any(ids>2160**3):
            return "Particle ID out of range"
        if sum(ids==0) > 1:
            return "Found multiple zero IDs"

        # Return None if everything looks ok
        return None


class SubTabFile(BinaryFile):
    """
    Class for reading Millennium-1 sub_tab files written by L-SubFind
    """
    def __init__(self, fname, *args):
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
        self.add_dataset("SubMostBoundID", np.int64,   (nsubgroups,))
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
            return "Sum of NSubPerHalo is wrong"
        ind = nsubperhalo > 0
        if np.any(firstsubofhalo[ind]<0) or np.any(firstsubofhalo[ind]>=nsubgroups):
            return "FirstSubOfHalo out of range"
        if np.any(suboffset<0) or np.any(suboffset+sublen>nids):
            return "Subhalo particle index(es) out of range"

        # Check subhalo parent group index
        parent = np.repeat(np.arange(ngroups, dtype=np.int32), nsubperhalo)
        if np.any(subparenthalo != parent):
            return "Subhalo parent halo index is wrong"

        # Check quantities which should be finite
        for name in ("Halo_M_Mean200",   "Halo_R_Mean200", 
                     "Halo_M_Crit200",   "Halo_R_Crit200", 
                     "Halo_M_TopHat200", "Halo_R_TopHat200",
                     "SubPos", "SubVel", "SubVelDisp",
                     "SubVmax", "SubSpin", "SubHalfMass"):
            data = self[name][...]
            if np.any(np.logical_not(np.isfinite(data))):
                return "Quantity %s has non-finite value" % name
        
        # Return None if everything looks ok
        return None


class SubIDsFile(BinaryFile):
    """
    Class for reading Millennium-1 sub_ids files written by L-SubFind
    """
    def __init__(self, fname, *args):
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

        # Read header
        Nids = self["Nids"][...]

        # Particle IDs
        self.add_dataset("groupIDs",   np.int64, (Nids,))

    def sanity_check(self):
        
        ids = self["groupIDs"][...]
        if np.any(ids < 0):
            return "Found negative ID"

        # Split ID into particle ID and hash key
        key = np.right_shift(ids, 34)
        ids = ids - np.left_shift(key, 34)

        # Note: hash table size and max ID hard coded here
        if np.any(key < 0) or np.any(key >= 256**3):
            return "Hash key out of range"
        if np.any(ids<0) or np.any(ids>2160**3):
            return "Particle ID out of range"
        if sum(ids==0) > 1:
            return "Found multiple zero IDs"

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
        self.add_dataset("PartType1/ParticleIDs", np.int64, (n,))
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
            return "Particle coordinate is not finite"
        if np.any(pos < 0.0) or np.any(pos > boxsize):
            return "Particle coordinate out of range"
        # (don't dealloc positions yet - needed for hash table check)

        # Check velocities
        vel = self["PartType1/Velocities"][...]
        if not(np.all(np.isfinite(vel))):
            return "Particle velocity is not finite"
        if np.any(abs(vel) > 1.0e6):
            return "Suspiciously high velocity"
        del vel

        # Check IDs
        ids = self["PartType1/ParticleIDs"][...]
        if np.any(ids < 0) or np.any(ids > nptot):
            return "Particle ID out of range"
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
                return "Particle not in correct hash cell"
        
        # Return None if everything looks ok
        return None
