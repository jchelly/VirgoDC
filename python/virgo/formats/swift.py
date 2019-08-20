#!/bin/env python

import re
import collections.abc

import h5py
import numpy
import unyt
import unyt.dimensions

# Unyt unit names for basic units in SWIFT
base_units = {
    "L" : unyt.cm,
    "M" : unyt.g,
    "t" : unyt.s,
    "T" : unyt.K,
    "I" : None, # TODO: determine what this should be!
}


class SwiftBaseWrapper(collections.abc.Mapping):
    """
    Base class for the Group, Dataset and File objects.
    Implements the (immutable) Mapping interface.

    This wraps the input object and forwards any attribute 
    or item access to the wrapped object.
    """
    def __init__(self, obj):
        self.obj = obj
        
    def __getattr__(self, name):
        return getattr(self.obj, name)
    
    def __getitem__(self, key):
        return self.obj[key]

    def __iter__(self):
        return self.obj.__iter__()
        
    def __len__(self):
        return self.obj.__len__()


class SwiftDataset(SwiftBaseWrapper):
    
    def __init__(self, obj):
        """
        This handles opening a dataset. We need to read the dataset
        specific metadata and also make sure to keep a copy of the
        simulation-wide metadata.
        """
        super(SwiftDataset, self).__init__(obj)

        # Read unit info for this dataset, if there is any
        self.unit_exponents = {}
        for unit in base_units:
            try:
                self.unit_exponents[unit] = self.obj.attrs["U_"+unit+" exponent"][0]
            except KeyError:
                pass

        # Read cosmology info for this dataset if present
        self.cosmology_exponents = {}
        try:
            self.a_exponent = self.obj.attrs["a-scale exponent"][0]
        except KeyError:
            self.a_exponent = None
        try:
            self.h_exponent = self.obj.attrs["h-scale exponent"][0]
        except KeyError:
            self.h_exponent = None

    def get_unit(self):

        # Construct unit for this dataset
        units = 1
        for unit, exponent in self.unit_exponents.items():
            if exponent != 0:
                cgs = self.base_units_cgs[unit]
                units *= (base_units[unit]**exponent) * (cgs**exponent)
        
        # Add cosmological factors
        if self.a_exponent is not None:
            units *= (unyt.Unit('a', registry=self.reg)**self.a_exponent)
        if self.h_exponent is not None:
            units *= (unyt.Unit('h', registry=self.reg)**self.h_exponent)

        return units

    def __getitem__(self, key):
        """
        This handles reading the actual data and returning a quantity
        with suitable units.
        """
        # Read the data
        data = self.obj[key]

        # Return a unyt.Quantity with suitable units
        result = unyt.array.unyt_array(data, self.get_unit(), registry=self.reg)

        # Attach a and h info
        result.a_exponent = self.a_exponent
        result.h_exponent = self.h_exponent

        return result


class SwiftGroup(SwiftBaseWrapper):
    """
    Class to wrap a h5py.Group. This needs to ensure that if
    we open a sub-group or dataset we return a wrapped group
    or dataset rather than the underlying h5py object.

    We also need to attach a copy of the unit metadata to the
    returned object.
    """
    def __init__(self, obj):
        super(SwiftGroup, self).__init__(obj)

    def __getitem__(self, key):
        obj = self.obj[key]
        if isinstance(obj, h5py.Dataset):
            result = SwiftDataset(obj)
        elif isinstance(obj, h5py.Group):
            result = SwiftGroup(obj)
        else:
            result = obj
        if result is not obj:
            result.cosmology = self.cosmology
            result.base_units_cgs = self.base_units_cgs
            result.reg = self.reg
        return result


class SwiftSnapshot(SwiftGroup):
    """
    This is a wrapper around the h5py.File object for an
    open snapshot file. Reads simulation-wide metadata on 
    initialisation.

    All arguments are passed to the underlying h5py.File.
    """
    def __init__(self, *args, **kwargs):
 
        # Open the HDF5 file
        super(SwiftSnapshot, self).__init__(h5py.File(*args, **kwargs))

        # Read unit information
        self.base_units_cgs = {}
        if "Units" in self.obj:
            for name in self.obj["Units"].attrs.keys():
                m = re.match(r"^.*\(U_(.)\)$", name)
                if m is not None:
                    self.base_units_cgs[m.group(1)] = self.obj["Units"].attrs[name][0]
        else:
            raise KeyError("Unable to find Units group in file %s" % self.obj.filename)

        # Read cosmology information required for unit conversions (if present)
        self.cosmology = {}
        if "Cosmology" in self.obj:
            try:
                self.cosmology["h"] = self.obj["Cosmology"].attrs["h"][0]
            except KeyError:
                pass
            try:
                self.cosmology["a"] = self.obj["Cosmology"].attrs["Scale-factor"][0]
            except KeyError:
                pass

        # Create a unit system containing h and a for this snapshot
        self.reg = unyt.UnitRegistry()
        self.reg.add("a", self.cosmology["a"], unyt.dimensions.dimensionless)
        self.reg.add("h", self.cosmology["h"], unyt.dimensions.dimensionless)

