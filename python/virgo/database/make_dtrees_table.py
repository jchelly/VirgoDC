#!/bin/env python
#
# Module to generate a database table from dhalo trees
#

import collections
import io

import numpy as np
import h5py

import virgo.database.index_trees as index_trees


def sql_type(dtype):
    """Given a numpy data type, return the corresponding SQL type"""
    if dtype.kind in ("i","u","f"):
        # It's a numeric type
        if dtype == np.int32:
            return "integer"
        elif dtype == np.int64:
            return "bigint"
        elif dtype == np.float32:
            return "real"
        elif dtype == np.float64:
            return "float"
        else:
            raise ValueError("Unsupported data type "+str(dtype))
    elif dtype.kind == "S":
        # It's a string
        # Note: this assumes 1 byte = 1 character!
        return ("char(%d)" % dtype.itemsize)
    else:
        # Not numeric or string, don't know what to do with this!
        raise ValueError("Unsupported data type "+str(dtype))   


def numpy_type(sqltype):
    """Given an SQL data type, return the corresponding numpy type"""
    m = re.match("char\(([0-9]+)\)", sqltype.strip())
    if m is not None:
        # It's a string
        return np.dtype("|S"+m.group(1))
    else:
        # It's a numeric type
        if sqltype == "integer" or sqltype == "int":
            return np.int32
        elif sqltype == "bigint":
            return np.int64
        elif sqltype == "real":
            return np.float32
        elif sqltype == "float":
            return np.float64
        else:
            raise ValueError("Unsupported data type "+sqltype)


def read_dhalo_trees(basename):
    """
    Read trees from a set of HDF5 files and return a
    dictionary of arrays with the subhalo properties.

    Input files are asssumed to have names which can be
    generated from "%s.%d.hdf5" % (basename, ifile).
    """
    
    # Read in the tree file(s)
    ifile  = 0
    nfiles = 1
    data   = collections.OrderedDict()
    while ifile < nfiles:
        treefile = h5py.File("%s.%d.hdf5" % (basename, ifile), "r")
        if ifile == 0:
            nfiles = treefile["fileInfo"].attrs["numberOfFiles"]
        for uname in treefile["haloTrees"].keys():
            name = str(uname)
            if ifile == 0:
                data[name] = []
            data[name].append(treefile["haloTrees"][name][...])
        treefile.close()
        ifile += 1
        
    # Combine arrays from separate files and return
    for name in data.keys():
        data[name] = np.concatenate(data[name], axis=0)
    return data


def add_depth_first_index(data):
    """
    Add depth first indexing information to the merger trees
    """

    # Calculate main branch mass
    data["mBranch"] = index_trees.find_progenitor_branch_mass_delucia(data["nodeIndex"], 
                                                                      data["descendantIndex"], 
                                                                      data["snapshotNumber"], 
                                                                      data["nodeMass"])
    # This numbers subhalos from 0 to (n-1) in depth first order
    data["depthFirst"], data["endMainBranch"], data["lastProgenitor"] = index_trees.depth_first_index(data["nodeIndex"], 
                                                                                                      data["descendantIndex"], 
                                                                                                      data["mBranch"])
    return data


def write_sql_server_native_file(tablename, data, data_file, script_file):
    """
    Write out an SQL server native file and the
    corresponding creat table script for the merger trees.

    Input data should be a dictionary of arrays with one entry
    for each table column. 3D vector quantities will be split into
    three columns with _x, _y, _z suffixes.
    """

    # Get number of objects
    n = data[data.keys()[0]].shape[0]

    # Generate numpy data type for one row:
    # To do this we need to make a list of (name, datatype) tuples
    # with one tuple per column.
    type_list = []
    for name in data.keys():
        if data[name].shape[0] != n:
            raise ValueError("All arrays must have the same size in first dimension")
        if len(data[name].shape) == 1:
            # Scalar quantity
            type_list.append((name, data[name].dtype))
        elif len(data[name].shape) == 2 and data[name].shape[1] == 3:
            # Vector quantity - split into three columns
            for suffix in ("x","y","z"):
                type_list.append((name+"_"+suffix, data[name].dtype))
        else:
            raise ValueError("Can't handle arrays with dimensions other than (n,) or (n,3)")
    row_dtype = np.dtype(type_list, align=False)
    ncols = len(row_dtype.fields)

    # Make array of this type and populate with data
    table = np.ndarray(n, dtype=row_dtype)
    for name in data.keys():
        if len(data[name].shape) == 1:
            table[name][:] = data[name][:]
        else:
            for i, suffix in enumerate(("x","y","z")):
                table[name+"_"+suffix][:] = data[name][:,i]
             
    # Write native data file
    outfile = open(data_file, "w")
    table.tofile(outfile)
    outfile.close()
   
    # Write create table script
    outfile = open(script_file, "w")
    outfile.write("create table %s (\n" % tablename)
    for name in row_dtype.names:
        sql_datatype = sql_type(row_dtype.fields[name][0])
        outfile.write("  %s %s NOT NULL,\n" % (name, sql_datatype))
    outfile.write(")\n")
    outfile.close()
    


def make_dtrees_table(tablename, input_basename, output_basename):
    """
    Given a set of HDF5 tree files specified by basename,
    generate SQL Server native file for input into a database
    and the corresponding create table script.
    """
    data = read_dhalo_trees(input_basename)
    add_depth_first_index(data)   
    write_sql_server_native_file(tablename, data, 
                                 output_basename+".dat", 
                                 output_basename+".sql")
    
