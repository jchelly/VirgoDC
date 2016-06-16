#!/bin/env python
#
# Check for missing files in an Eagle run
#

import sys
import os
import numpy as np
import h5py
from glob import glob
from collections import defaultdict

def eagle_filename(simdir, dirbase, filebase, isnap, ifile, cache={}):
    """
    Generate name of a file in an Eagle output

    Raises IOError if the directory does not exist because
    then we can't determine the redshift string.
    """
    key = (simdir, dirbase, isnap)
    if key in cache:
        zstring = cache[key]
    else:
        pattern  = "%s/%s_%03d_z???p???" % (simdir, dirbase, isnap)
        dirnames = glob(pattern)
        if len(dirnames) == 0:
            raise IOError("Unable to find directory %s" % pattern)
        elif len(dirnames) > 1:
            raise IOError("Found multiple directories %s" % pattern)
        else:
            dirname = dirnames[0]
        zstring = dirname[-8:]
        cache[key] = zstring

    return "%s/%s_%03d_%s/%s_%03d_%s.%d.hdf5" % (simdir, dirbase, isnap, zstring, 
                                                 filebase, isnap, zstring, ifile)
 

def check_dir(basedir, dirbase, filebase, nfiles_attr, isnap):
    
    files_ok = 0
    nfiles   = -1
    try:
        # Try to open first file to determine number of files
        fname = eagle_filename(basedir, dirbase, filebase, isnap, 0)
        f = h5py.File(fname, "r")
        nfiles = f["Header"].attrs[nfiles_attr]
        f.close()
    except IOError:
        pass
    else:
        # Get directory listing to check presence of remaining files -
        # this is FAR faster than opening them all but doesn't check
        # that they're readable or even that they're files.
        # But that's ok if we're going to check their hashes later.
        dirname = os.path.dirname(fname)
        try:
            fnames  = os.listdir(dirname)
        except OSError:
            pass
        else:
            for ifile in range(nfiles):
                fname = eagle_filename(basedir, dirbase, filebase, isnap, ifile)
                if os.path.basename(fname) in fnames:
                    files_ok += 1
    if files_ok == nfiles:
        print "%20s %03d: OK      - %d of %d %s files present" % (dirbase, isnap, files_ok, nfiles, filebase)
        return True
    elif nfiles == -1:
        print "%20s %03d: MISSING - no %s files present!" % (dirbase, isnap, filebase)
        return False
    else:
        print "%20s %03d: MISSING - %d of %d %s files present" % (dirbase, isnap, files_ok, nfiles, filebase)
        return False


def check_eagle_run(basedir, num_snaps, num_snips):
    """
    Check for presence of output files in an Eagle run
    """

    count_missing  = defaultdict(lambda : 0)
    count_expected = defaultdict(lambda : 0)

    for outtype in ("snap", "snip"):
    
        if outtype == "snap":
            num = num_snaps
        else:
            num = num_snips
        
        for isnap in range(num):

            # Check sn[ia]pshot data
            if not(check_dir(basedir, outtype+"shot", outtype, "NumFilesPerSnapshot", isnap)):
                count_missing[outtype] += 1
            count_expected[outtype] += 1

            # Check FoF outputs
            if outtype == "snap":
                dirbase  = "groups"
                filebase = "group_tab"
            else:
                dirbase  = "groups_snip"
                filebase = "group_snip_tab"
            if not(check_dir(basedir, dirbase, filebase, "NTask", isnap)):
                count_missing[filebase] += 1
            count_expected[filebase] += 1

            # Check subfind particle data
            if outtype == "snap":
                dirbase  = "particledata"
                filebase = "eagle_subfind_particles"
            else:
                dirbase  = "particledata_snip"
                filebase = "eagle_subfind_snip_particles"
            if not(check_dir(basedir, dirbase, filebase, "NumFilesPerSnapshot", isnap)):
                count_missing[filebase] += 1
            count_expected[filebase] += 1

            # Check subfind catalogues
            if outtype == "snap":
                dirbase  = "groups"
                filebase = "eagle_subfind_tab"
            else:
                dirbase  = "groups_snip"
                filebase = "eagle_subfind_snip_tab"            
            if not(check_dir(basedir, dirbase, filebase, "NTask", isnap)):
                count_missing[filebase] += 1
            count_expected[filebase] += 1

    # Print summary
    print
    print "Summary of missing or incomplete directories:"
    print
    for outtype in count_expected.keys():
        if count_missing[outtype] > 0:
            print "%30s - %d of %d directories not complete" % (outtype, 
                                                                count_missing[outtype],
                                                                count_expected[outtype])

if __name__ == "__main__":

    basedir   = sys.argv[1]
    num_snaps = int(sys.argv[2])
    num_snips = int(sys.argv[3])

    check_eagle_run(basedir, num_snaps, num_snips)
