#!/bin/env python
#
# Check for missing files in a P-Gadget3 run
#

import sys
import os
import numpy as np
import h5py
from glob import glob
from collections import defaultdict
from virgo.formats import gadget_snapshot, subfind_pgadget3

def pgadget3_filename(simdir, dirbase, filebase, isnap, ifile):
    """
    Generate name of a file in a P-Gadget3 simulation
    """
    if filebase == "500_dm": # special case for Millennium WMAP7
        return "%s/%s_%03d/%s_%03d.%d.hdf5" % (simdir, dirbase, isnap, filebase, isnap, ifile)
    else:
        return "%s/%s_%03d/%s_%03d.%d" % (simdir, dirbase, isnap, filebase, isnap, ifile)



def check_dir(basedir, dirbase, filebase, isnap, get_nfiles_func):
    
    files_ok = 0
    nfiles   = -1
    try:
        # Try to open first file to determine number of files
        fname = pgadget3_filename(basedir, dirbase, filebase, isnap, 0)
        nfiles = get_nfiles_func(fname)
    except Exception:
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
                fname = pgadget3_filename(basedir, dirbase, filebase, isnap, ifile)
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


def check_pgadget3_run(basedir, snapbase, max_num, first_num, step):
    """
    Check for presence of output files in an Eagle run
    """

    count_missing  = defaultdict(lambda : 0)
    count_expected = defaultdict(lambda : 0)
    
    for isnap in range(first_num, max_num+1, step):

        # Check snapshot data
        get_nfiles_func = lambda fname : gadget_snapshot.open(fname)["Header"].attrs["NumFilesPerSnapshot"]
        if not(check_dir(basedir, "snapdir", snapbase, isnap, get_nfiles_func)):
            count_missing["snapshot"] += 1
        count_expected["snapshot"] += 1

        # Check sub_tab files
        get_nfiles_func = lambda fname : subfind_pgadget3.GroupTabFile(fname)["NTask"][...]
        if not(check_dir(basedir, "groups", "group_tab", isnap, get_nfiles_func)):
            count_missing["group_tab"] += 1
        count_expected["group_tab"] += 1
        
        # Check sub_ids files
        get_nfiles_func = lambda fname : subfind_pgadget3.GroupIDsFile(fname)["NTask"][...]
        if not(check_dir(basedir, "groups", "group_ids", isnap, get_nfiles_func)):
            count_missing["group_ids"] += 1
        count_expected["group_ids"] += 1
   
        # Check sub_tab files
        get_nfiles_func = lambda fname : subfind_pgadget3.SubTabFile(fname)["NTask"][...]
        if not(check_dir(basedir, "groups", "subhalo_tab", isnap, get_nfiles_func)):
            count_missing["subhalo_tab"] += 1
        count_expected["subhalo_tab"] += 1
        
        # Check sub_ids files
        get_nfiles_func = lambda fname : subfind_pgadget3.SubIDsFile(fname)["NTask"][...]
        if not(check_dir(basedir, "groups", "subhalo_ids", isnap, get_nfiles_func)):
            count_missing["subhalo_ids"] += 1
        count_expected["subhalo_ids"] += 1
   

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

    basedir  = sys.argv[1]
    snapbase = sys.argv[2]
    max_num  = int(sys.argv[3])

    if len(sys.argv) == 6:
        first_num = int(sys.argv[4])
        step      = int(sys.argv[5])
    elif len(sys.argv) == 4:
        first_num = 0
        step      = 1
    else:
        print "Incorrect number of arguments!"
        sys.exit(1)

    check_pgadget3_run(basedir, snapbase, max_num, first_num, step)
