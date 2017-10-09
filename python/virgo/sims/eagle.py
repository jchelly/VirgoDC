#!/bin/env python

import glob
import re

def eagle_filename(basedir, basename, isnap, ifile, cache={}):
    """
    Generate the path to an Eagle output file.
    Basename determines the file type and must
    be one of the keys in dir_names, below.
    """

    dir_names = {
        "eagle_subfind_tab"            : "groups",
        "eagle_subfind_snip_tab"       : "groups_snip",
        "group_tab"                    : "groups",
        "group_snip_tab"               : "groups_snip",
        "eagle_subfind_particles"      : "particledata",
        "eagle_subfind_snip_particles" : "particledata_snip",
        "snapshot"                     : "snapshot",
        "snipshot"                     : "snipshot",
        }
    dir_name = dir_names[basename]

    key = (basedir, basename, isnap)
    if key in cache:
        # We already know the redshift label for this sn[ia]pshot
        zstr  = cache[key]
        fname = ("%s/%s_%03d_%s/%s_%03d_%s.%d.hdf5" % 
                 (basedir, dir_name, isnap, zstr, basename, isnap, zstr, ifile))
    else:
        # Don't know the redshift label, so need to access filesystem with glob
        pattern = ("%s/%s_%03d_z???p???/%s_%03d_z???p???.%d.hdf5" % 
                   (basedir, dir_name, isnap, basename, isnap, ifile))
        fname = glob.glob(pattern)
        if len(fname) != 1:
            raise IOError("Unable to find single file matching %s" % pattern)
        else:
            fname = fname[0]
        m = re.match(r"^.*(z[0-9][0-9][0-9]p[0-9][0-9][0-9])\.[0-9]+\.hdf5$", fname)
        cache[key] = m.group(1)

    return fname

