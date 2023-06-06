#!/bin/env python

import re

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping    

import h5py
import numpy as np
import unyt
import unyt.dimensions as dim

# Unyt unit names for basic units in SWIFT
base_units = {
    "L" : unyt.cm,
    "M" : unyt.g,
    "t" : unyt.s,
    "T" : unyt.K,
    "I" : None, # TODO: determine what this should be!
}


def unit_registry_from_snapshot(snap):

    # Read snapshot metadata
    physical_constants_cgs = {name : float(value) for name, value in snap["PhysicalConstants/CGS"].attrs.items()}
    cosmology = {name : float(value) for name, value in snap["Cosmology"].attrs.items()}
    a = unyt.unyt_quantity(cosmology["Scale-factor"])
    h = unyt.unyt_quantity(cosmology["h"])

    # Create a new registry
    reg = unyt.unit_registry.UnitRegistry()

    # Define code and snapshot base units
    for group_name, prefix in (("Units", "snap"),
                               ("InternalCodeUnits", "code")):
        units_cgs = {name : float(value) for name, value in snap[group_name].attrs.items()}
        unyt.define_unit(prefix+"_length",      units_cgs["Unit length in cgs (U_L)"]*unyt.cm,     registry=reg)
        unyt.define_unit(prefix+"_mass",        units_cgs["Unit mass in cgs (U_M)"]*unyt.g,        registry=reg)
        unyt.define_unit(prefix+"_time",        units_cgs["Unit time in cgs (U_t)"]*unyt.s,        registry=reg)
        unyt.define_unit(prefix+"_temperature", units_cgs["Unit temperature in cgs (U_T)"]*unyt.K, registry=reg)
        unyt.define_unit(prefix+"_angle",       1.0*unyt.rad,                                      registry=reg)
        unyt.define_unit(prefix+"_current",     units_cgs["Unit current in cgs (U_I)"]*unyt.A,     registry=reg)

    # Add the expansion factor as a dimensionless "unit"
    unyt.define_unit("a", a, dim.dimensionless, registry=reg)
    unyt.define_unit("h", h, dim.dimensionless, registry=reg)

    # Create a new unit system using the snapshot units as base units
    us = unyt.UnitSystem(
        "snap_units",
        unyt.Unit("snap_length", registry=reg),
        unyt.Unit("snap_mass", registry=reg),
        unyt.Unit("snap_time", registry=reg),
        unyt.Unit("snap_temperature", registry=reg),
        unyt.Unit("snap_angle", registry=reg),
        unyt.Unit("snap_current", registry=reg),
        registry=reg
    )
    
    # Create a registry using this base unit system
    reg = unyt.unit_registry.UnitRegistry(lut=reg.lut, unit_system=us)

    # Add some units which might be useful for dealing with VR data
    unyt.define_unit("swift_mpc",  1.0e6*physical_constants_cgs["parsec"]*unyt.cm, registry=reg)
    unyt.define_unit("swift_msun", physical_constants_cgs["solar_mass"]*unyt.g, registry=reg)
    unyt.define_unit("newton_G", physical_constants_cgs["newton_G"]*unyt.cm**3/unyt.g/unyt.s**2, registry=reg)

    return reg


def units_from_attributes(attrs, registry):
    """
    Create a unyt.Unit object from dataset attributes

    attrs: the SWIFT dataset attributes dict
    registry: unyt unit registry with a, h and unit system for the snapshot

    Returns a unyt Unit object.
    """
    # Determine unyt unit for this quantity
    u = unyt.dimensionless
    unit_system = registry.unit_system
    base = registry.unit_system.base_units
    for symbol, baseunit in (("I", base[dim.current_mks]),
                             ("L", base[dim.length]),
                             ("M", base[dim.mass]),
                             ("T", base[dim.temperature]),
                             ("t", base[dim.time])):
        unit = unyt.Unit(baseunit, registry=registry)
        exponent = attrs["U_%s exponent" % symbol][0]
        if exponent == 1.0:
            if u is unyt.dimensionless:
                u = unit
            else:
                u = u*unit
        elif exponent != 0.0:
            if u is unyt.dimensionless:
                u = unit**exponent
            else:
                u = u*(unit**exponent)

    # Add expansion factor
    a_scale_exponent = attrs["a-scale exponent"][0]
    a_unit = unyt.Unit("a", registry=registry)**a_scale_exponent
    if a_scale_exponent != 0:
        if u is unyt.dimensionless:
            u = a_unit
        else:
            u = u*a_unit

    # Add h factor
    h_scale_exponent = attrs["h-scale exponent"][0]
    h_unit = unyt.Unit("h", registry=registry)**h_scale_exponent
    if h_scale_exponent != 0:
        if u is unyt.dimensionless:
            u = h_unit
        else:
            u = u*h_unit

    unit = unyt.Unit(u, registry=registry)

    # SOAP outputs can have units which are not just powers of the base units.
    cgs_conversion_from_attrs = float(attrs["Conversion factor to CGS (including cosmological corrections)"])
    cgs_conversion_from_unyt = float((1.0*unit).in_cgs().value)
    factor = cgs_conversion_from_attrs / cgs_conversion_from_unyt
    if np.isclose(factor, 1.0, atol=0.0, rtol=1.0e-9):
        return unit
    else:
        return np.round(factor, decimals=9)*unit


class SwiftBaseWrapper(Mapping):
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
        specific metadata attributes and store them.
        """
        super(SwiftDataset, self).__init__(obj)

        self.attrs = {}
        for name, value in self.obj.attrs.items():
            self.attrs[name] = value

    @property
    def units(self):
        return units_from_attributes(self.attrs, self.registry)
    
    def __getitem__(self, key):
        """
        This handles reading the actual data and returning a unyt quantity
        with suitable units.
        """
        data = self.obj[key]
        return unyt.array.unyt_array(data, self.units, registry=self.registry)


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
            result.registry = self.registry
        return result

    def get_unit(self, name):
        """
        Given a unit name, return a Unit object
        """
        return unyt.Unit(name, registry=self.registry)


class SwiftSnapshot(SwiftGroup):
    """
    This is a wrapper around the h5py.File object for an
    open snapshot file.

    All arguments are passed to the underlying h5py.File.
    """
    def __init__(self, *args, **kwargs):
 
        # Open the HDF5 file
        super(SwiftSnapshot, self).__init__(h5py.File(*args, **kwargs))

        # Read unit information
        self.registry = unit_registry_from_snapshot(self.obj)
