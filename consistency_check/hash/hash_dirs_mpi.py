#!/bin/env python

import sys
import os
import os.path
import hashlib
import socket
import datetime

# Need this to write database with hashes
try:
    import sqlite3 as sqldb
except ImportError:
    sqldb = None

# Need this module to set extended attributes on files
try:
    from xattr import xattr
except ImportError:
    xattr = None

from mpi4py import MPI
import traceback
import argparse
from fnmatch import fnmatch

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

dirlist = None

class DirectoryList:
    """
    Class to assign directories for hashing to MPI tasks.
    Also stores results acccumulated from all tasks.
    """
    def __init__(self, basedir, dbfile, hash_type, resume):
        """
        Initialize full list of sub-directories on rank 0
        """
        self.cache = {}
        if comm_rank == 0:
            self.dirs         = os.walk(basedir, followlinks=True)
            self.num_finished = 0
            self.basedir      = basedir
            self.hash_type    = hash_type
            self.all_hashes   = {}
            self.cache        = {}
            if dbfile is not None:
                # Connect to database, create if necessary
                self.con = sqldb.connect(dbfile)
                try:
                    self.con.execute("create table hashes (basedir text, path text, hash_type text, hash text)")
                    self.con.commit()
                except sqldb.OperationalError:
                    pass
                if resume:
                    # Read hashes from database if resuming
                    cur = self.con.execute("select path, hash from hashes where basedir=? and hash_type=?", (basedir, hash_type))
                    for path, this_hash in cur:
                        self.cache[path] = this_hash
                    cur.close()
                else:
                    # Otherwise truncate hashes table
                    self.con.execute("delete from hashes")
                    self.con.commit()
            else:
                self.con = None
        self.cache = comm.bcast(self.cache)
        if resume and len(self.cache) == 0:
            raise Exception("Unable to read any hashes from database while trying to resume!")

    def __del__(self):
        if comm_rank == 0:
            if self.con is not None:
                self.con.close()

    def store_hashes(self, hashes):
        """
        Accumulate the supplied hashes on task zero.
        """
        assert comm_rank == 0
        if hashes is not None:
            self.all_hashes.update(hashes)
            if self.con is not None:
                for name, val in hashes.iteritems():
                    # Don't store anything if file could not be read or hash was read from database 
                    if name not in self.cache and val is not None:
                        self.con.execute("insert into hashes values(?,?,?,?)", (self.basedir, name, self.hash_type, val))
                self.con.commit()

    def request_task(self, hashes):
        """
        Request a directory to process, or return None if there are no more.
        Can be called from any task. Also returns result from previous
        directory to task zero.
        
        hashes is a dictionary with the results from the previous directory,
        or None if this is the first one requested.
        """
        if comm_rank > 0:
            comm.send(hashes, dest=0)
            return comm.recv(source=0)
        else:
            self.store_hashes(hashes)
            return next(self.dirs, None)

    def process_requests(self):
        """
        Respond to any incoming requests for tasks.
        Should only be called on task zero.

        Returns true while other tasks are still working.
        """
        assert comm_rank == 0
        status = MPI.Status()
        while True:
            if comm.iprobe(status=status):
                # Identify source of request
                source = status.Get_source()
                # Receive and store hashes of last directory processed, if any 
                self.store_hashes(comm.recv(source=source))
                # Send back the next directory, or None if there isn't one
                next_dir = next(self.dirs, None)
                comm.send(next_dir, dest=source)
                if next_dir is None:
                    self.num_finished += 1
            else:
                break
        return self.num_finished < comm_size - 1        


def hash_file(fname, hash_type):
    """
    Compute hash of the specified file
    """
    buffer_size = 10*1024*1024
    m = hashlib.new(hash_type)
    f = open(fname, "rb")
    while True:
        # Task zero needs to process any pending requests from other tasks
        if comm_rank == 0:
            dirlist.process_requests()
        # Read next block of data
        data = f.read(buffer_size)
        # Update hash
        if data != "":
            m.update(data)
        else:
            break

    f.close()
    return m.hexdigest()


def hash_dir(basedir, dirpath, dirnames, filenames, 
             per_dir_file, all_dirs_file, 
             set_attrs, hash_type, include, exclude):
    """
    Hash all files in list filenames in directory dirpath
    """

    # Compute hashes for all files in this directory
    hashes = {}
    hashes_with_path = {}
    for name in filenames:
        full_name = os.path.normpath(os.path.join(dirpath, name))
        rel_path  = os.path.normpath(os.path.relpath(full_name, basedir))
        # Avoid hashing our own output file(s)
        is_out_file = False
        if all_dirs_file is not None and rel_path == os.path.normpath(os.path.relpath(all_dirs_file, basedir)):
            is_out_file = True
        if per_dir_file is not None and os.path.normpath(name) == os.path.normpath(per_dir_file):
            is_out_file = True
        if not(is_out_file):
            # Check if we should do this one according to include / exclude rules
            if include is None:
                do_file = True
            else:
                do_file = any([fnmatch(rel_path, os.path.normpath(inc)) for inc in include])
            if exclude is not None:
                if any([fnmatch(rel_path, os.path.normpath(exc)) for exc in exclude]):
                    do_file = False
            if do_file:
                if rel_path in dirlist.cache:
                    this_hash = dirlist.cache[rel_path]
                    cached = True
                else:
                    try:
                        # Calculate hash
                        this_hash = hash_file(full_name, hash_type)
                    except IOError:
                        # Failed - file may be unreadable, truncated etc
                        this_hash = None
                    cached = False
                # Store hash
                hashes[name] = this_hash
                hashes_with_path[rel_path] = this_hash
                # Add extended attribute
                if set_attrs and this_hash is not None:
                    attr = xattr(full_name)
                    attr["user.checksum."+hash_type] = this_hash
                if cached:
                    print "Task %4d : %s %s (Cached)" % (comm_rank, this_hash if this_hash is not None else "UNREADABLE", rel_path)
                else:
                    print "Task %4d : %s %s" % (comm_rank, this_hash if this_hash is not None else "UNREADABLE", rel_path)
        else:
            print "Task %4d : skipping output file %s" % (comm_rank, rel_path)

    # Get sorted list of names
    sorted_names = sorted(hashes.keys())

    if per_dir_file is not None:
        # Write a file with the hashes for this directory
        hashes_file = os.path.join(dirpath, per_dir_file)
        f = open(hashes_file, "w")
        f.write("# Written at %s UTC on host %s\n" % (datetime.datetime.utcnow().isoformat(), socket.gethostname()))
        f.write("# Hash type: %s\n" % (hash_type,))
        f.write("# Base directory: %s\n" % (os.path.abspath(basedir),))
        f.write("# This directory: %s\n" % (os.path.relpath(dirpath, basedir),))
        for name in sorted_names:
            f.write("%s %s\n" % (hashes[name] if hashes[name] is not None else "UNREADABLE", name))
        f.close()

    # Return dictionary with hash for each path
    return hashes_with_path


def recursive_hash(basedir, include, exclude,
                   per_dir_file=None, all_dirs_file=None, set_attrs=False,
                   hash_type='md5', dbfile=None, resume=False):
    """
    Hash files in all directories under basedir
    """

    # Check if we have and/or need xattr
    if set_attrs and xattr is None:
        raise Exception("Unable to import xattrs module")

    # Dictionary to store hashes of all files
    all_hashes = {}

    # Get full list of subdirectories and files
    global dirlist
    dirlist = DirectoryList(basedir, dbfile, hash_type, resume)

    # Process directories until we're done
    hashes = None
    while True:
        dir_to_do = dirlist.request_task(hashes)
        if dir_to_do is not None:
            (dirpath, dirnames, filenames) = dir_to_do
            hashes = hash_dir(basedir, dirpath, dirnames, filenames, 
                              per_dir_file, all_dirs_file, set_attrs, 
                              hash_type, include, exclude)
        else:
            break

    # Task zero needs to process requests until all other tasks finish
    if comm_rank == 0:
        while dirlist.process_requests():
            pass

    # Write output file with all hashes
    if all_dirs_file is not None:
        if comm_rank == 0:
            sorted_names = sorted(dirlist.all_hashes.keys())
            f = open(all_dirs_file, "w")
            f.write("# Written at %s UTC on host %s\n" % (datetime.datetime.utcnow().isoformat(), socket.gethostname()))
            f.write("# Hash type: %s\n" % (hash_type,))
            f.write("# Base directory: %s\n" % (os.path.abspath(basedir),))
            for name in sorted_names:
                f.write("%s %s\n" % (dirlist.all_hashes[name] if dirlist.all_hashes[name] is not None else "UNREADABLE", name))
            f.close()


if __name__ == "__main__":

    try:
        # Get command line parameters
        if comm_rank == 0:
            parser = argparse.ArgumentParser(description='Calculate hashes for all files under basedir')
            parser.add_argument('basedir',   help='Top of directory tree to be hashed')
            parser.add_argument('--outfile', help='Name of output file for hashes of all subdirectories')
            parser.add_argument('--dirfile', help='Name of output file to be placed in each subdirectory')
            parser.add_argument('--attrs', action='store_true', help='Add hashes to files as extended attributes')
            parser.add_argument('--type',    help='Specify hash algorithm to use', default='md5')
            parser.add_argument('--include', help='Only include files whose path relative to basedir matches one of the supplied patterns (comma separated)')
            parser.add_argument('--exclude', help='Exclude files whose path relative to basedir matches one of the supplied patterns (comma separated, overrides --include)')
            parser.add_argument('--database',help='Write hashes to the specified sqlite3 database file, which will be overwritten if --resume is not specified.')
            parser.add_argument('--resume', action="store_true", help='Resume an incomplete run by reading hashes from the database file instead of recomputing them.')
            try:
                args = parser.parse_args()
            except SystemExit:
                # Invalid parameters - need to MPI_Finalize rather than just exit
                args = None
        else:
            args = None
        args = comm.bcast(args)
        if args is not None:
            # Get lists of include / exclude patterns
            if args.include is not None:
                include = args.include.split(",")
            else:
                include = None
            if args.exclude is not None:
                exclude = args.exclude.split(",")
            else:
                exclude = None
            # Hash the files
            recursive_hash(args.basedir, include, exclude, 
                           per_dir_file=args.dirfile,
                           all_dirs_file=args.outfile,
                           set_attrs=args.attrs,
                           hash_type=args.type,
                           dbfile=args.database,
                           resume=args.resume)
    except (Exception, KeyboardInterrupt) as e:
        # Ensure all processes get killed if anything goes wrong
        sys.stderr.write("\n\n*** EXCEPTION ***\n"+str(e)+" on rank "+str(comm_rank)+"\n\n")
        traceback.print_exc(file=sys.stdout)
        sys.stderr.write("\n\n")
        sys.stderr.flush()
        comm.Abort()
    else:
        MPI.Finalize()
