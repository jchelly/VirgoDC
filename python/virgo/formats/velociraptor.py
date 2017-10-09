#!/bin/env python
#
# Classes for reading Velociraptor (binary) output
#

import os
import numpy as np
from virgo.util.read_binary import BinaryFile


class PropertiesFile(BinaryFile):
    """
    Class for reading *.properties files from Velociraptor
    """
    def __init__(self, fname, ikeepfof=0, 
                 GASON=False, STARON=False, BHON=False, HIGHRES=False, 
                 *args):
        BinaryFile.__init__(self, fname, *args)
        
        # Define header
        self.add_attribute("Header/ThisTask", np.int32)
        self.add_attribute("Header/NProcs",   np.int32)
        self.add_attribute("Header/ng",       np.uint64) # "unsigned long" in code - assuming 64 bit system here
        self.add_attribute("Header/ngtot",    np.uint64)
        self.add_attribute("Header/hsize",    np.int)
        hsize = self["Header"].attrs["hsize"]
        for i in range(hsize):
            self.add_attribute("Header/entry%03d" % i, np.dtype("S40"))

        # Define type to store each object
        type_list = [
            ("haloid",  np.uint64),
            ("ibound",  np.uint64),
            ("hostid",  np.uint64),
            ("numsubs", np.uint64),
            ("num",     np.uint64),
            ("stype",   np.uint32),
            ]
        if ikeepfof != 0:
            type_list += [("directhostid", np.uint64),
                          ("hostfofid",    np.uint64)]
        type_list += [
            ("gMvir",  np.float64),
            ("gcm",    np.float64, (3,)),
            ("gpos",   np.float64, (3,)),
            ("gcmvel", np.float64, (3,)),
            ("gvel",   np.float64, (3,)),
            ("gmass",  np.float64),
            ("gMFOF",  np.float64),
            ("gM200m", np.float64),
            ("gM200c", np.float64),
            ("gMvir_again",  np.float64),
            ("Efrac",  np.float64),
            ("gRvir",  np.float64),
            ("gsize",  np.float64),
            ("gR200m", np.float64),
            ("gR200c", np.float64),
            ("gRvir_again",  np.float64),
            ("gRhalfmass", np.float64),
            ("gRmaxvel",   np.float64),
            ("gmaxvel",    np.float64),
            ("gsigma_v",   np.float64),
            ("gveldisp",   np.float64, (3,3)),
            ("glambda_B",  np.float64),
            ("gJ",         np.float64, (3,)),
            ("gq",         np.float64),
            ("gs",         np.float64),
            ("geigvec",    np.float64, (3,3)),
            ("cNFW",       np.float64),
            ("Krot",       np.float64),
            ("T",          np.float64),
            ("Pot",        np.float64),
            ("RV_sigma_v", np.float64),
            ("RV_veldisp", np.float64, (3,3)),
            ("RV_lambda_B", np.float64),
            ("RV_J",       np.float64, (3,)),
            ("RV_q",       np.float64),
            ("RV_s",       np.float64),
            ("RV_eigvec",  np.float64, (3,3)),
            ]

        if GASON or STARON or BHON or HIGHRES:
            raise NotImplementedError("Runs with GASON?BHON/STARON/HIGHRES not supported!")

        halo_t = np.dtype(type_list, align=False)
        self.add_dataset("Halo", halo_t, (self["Header"].attrs["ng"],))
