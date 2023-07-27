#!/bin/env python
#
# Classes for reading Rockstar (binary) output
#

import os
import numpy as np
from virgo.util.read_binary import BinaryFile
from virgo.util.exceptions  import SanityCheckFailedException


class HalosFile(BinaryFile):
    """
    Class for reading halos*.bin files from Rockstar

    Use hydro=True to read output from rockstar-galaxies
    """
    def __init__(self, fname, hydro=False, *args):
        super(HalosFile, self).__init__(fname, *args)
        
        # Define header
        self.add_attribute("Header/magic",  np.uint64)
        self.add_attribute("Header/snap",   np.int64)
        self.add_attribute("Header/chunk",  np.int64)
        self.add_attribute("Header/scale",  np.float32)
        self.add_attribute("Header/om",     np.float32)
        self.add_attribute("Header/ol",     np.float32)
        self.add_attribute("Header/h0",     np.float32)
        self.add_attribute("Header/bounds", np.float32, (6,))
        self.add_attribute("Header/num_halos",        np.int64)
        self.add_attribute("Header/num_particles",    np.int64)
        self.add_attribute("Header/box_size",         np.float32)
        self.add_attribute("Header/particle_mass",    np.float32)
        self.add_attribute("Header/particle_type",    np.int64)
        self.add_attribute("Header/format_revision",  np.int32)
        self.add_attribute("Header/rockstar_version", np.dtype("S12"))
        self.add_attribute("Header/unused", np.dtype("S144"))
        assert (self.offset == 256)
                
        num_halos     = self["Header"].attrs["num_halos"][()]
        num_particles = self["Header"].attrs["num_particles"][()]
        format_revision = self["Header"].attrs["format_revision"]

        # Halo catalogue is stored as an array of structures
        fields = [
            ("id",        np.int64),
            ("pos",       np.float32, (6,)), 
            ("corevel",   np.float32, (3,)), 
            ("bulkvel",   np.float32, (3,)), 
            ("m",         np.float32),
            ("r",         np.float32),
            ("child_r",   np.float32),
            ("vmax_r",    np.float32),
            ("mgrav",     np.float32),
            ("vmax",      np.float32),
            ("rvmax",     np.float32),
            ("rs",        np.float32),
            ("klypin_rs", np.float32),
            ("vrms",      np.float32),
            ("J",         np.float32, (3,)),
            ("energy",    np.float32),
            ("spin",      np.float32),
            ("alt_m",     np.float32, (4,)),
            ("Xoff",      np.float32),
            ("Voff",      np.float32),
            ("b_to_a",    np.float32),
            ("c_to_a",    np.float32),
            ("A",         np.float32, (3,)),
            ("b_to_a2",   np.float32),
            ("c_to_a2",   np.float32),
            ("A2",        np.float32, (3,)),
            ("bullock_spin",    np.float32),
            ("kin_to_pot",      np.float32),
            ("m_pe_b",          np.float32),
            ("m_pe_d",          np.float32),
        ]

        # Half mass radius appears to be missing for format_revision=1
        if format_revision >= 2:
            fields += [("halfmass_radius", np.float32),]

        fields += [
            ("num_p",               np.int64),
            ("num_child_particles", np.int64),
            ("p_start",             np.int64),
            ("desc",                np.int64),
            ("flags",               np.int64),
            ("n_core",              np.int64),
            ("min_pos_err",         np.float32),
            ("min_vel_err",         np.float32),
            ("min_bulkvel_err",     np.float32),
        ]

        # Additional fields present in rockstar-galaxies branch
        if hydro:
            fields += [
                ("type", np.int32),
                ("sm", np.float32),
                ("gas", np.float32),
                ("bh", np.float32),
                ("peak_density", np.float32),
                ("av_density", np.float32),
            ]

        halo_t = np.dtype(fields, align=True)
        self.add_dataset("Halo", halo_t, (num_halos,))
        
        # Then we have the IDs of particles in halos
        self.add_dataset("IDs", np.int64, (num_particles,))

    def sanity_check(self):
        """
        Quick sanity check: indexes of particles in halo should be valid
        indexes into the ID array.

        Meant to catch the case where we got the halo_t definition wrong
        or the hydro parameter is set incorrectly.
        """
        halo = self["Halo"][...]
        p_start = halo["p_start"]
        num_p = halo["num_p"]
        if not np.all(p_start + num_p <= self["IDs"].shape[0]):
            raise SanityCheckFailedException("Particle index too large")
        if np.any(p_start < 0):
            raise SanityCheckFailedException("Particle index is negative")
