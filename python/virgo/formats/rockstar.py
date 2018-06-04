#!/bin/env python
#
# Classes for reading Rockstar (binary) output
#

import os
import numpy as np
from collections import Mapping
from virgo.util.read_binary import BinaryFile

class HalosFile(BinaryFile):
    """
    Class for reading halos*.bin files from Rockstar
    """
    def __init__(self, fname, *args):
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

        # Halo catalogue is stored as an array of structures
        halo_t = np.dtype([
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
                ("halfmass_radius", np.float32),
                ("num_p",               np.int64),
                ("num_child_particles", np.int64),
                ("p_start",             np.int64),
                ("desc",                np.int64),
                ("flags",               np.int64),
                ("n_core",              np.int64),
                ("min_pos_err",         np.float32),
                ("min_vel_err",         np.float32),
                ("min_bulkvel_err",     np.float32),
                ], align=True)
        self.add_dataset("Halo", halo_t, (num_halos,))
        
        # Then we have the IDs of particles in halos
        self.add_dataset("IDs", np.int64, (num_particles,))
