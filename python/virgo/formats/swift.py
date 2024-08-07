#!/bin/env python

import re
import io

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping    

import h5py
import numpy as np
import unyt
import unyt.dimensions as dim
import swiftsimio


# Unyt unit names for basic units in SWIFT
base_units = {
    "L" : unyt.cm,
    "M" : unyt.g,
    "t" : unyt.s,
    "T" : unyt.K,
    "I" : None, # TODO: determine what this should be!
}


def soap_unit_registry_from_snapshot(snap):
    """
    Generate system of units as used in SOAP from SWIFT metadata
    """
    # Read snapshot metadata
    physical_constants_cgs = {name : float(value) for name, value in snap["PhysicalConstants/CGS"].attrs.items()}
    cosmology = {name : float(value) for name, value in snap["Cosmology"].attrs.items()}
    a = unyt.unyt_quantity(cosmology["Scale-factor"])
    h = unyt.unyt_quantity(cosmology["h"])

    # Create a new registry
    reg = unyt.unit_registry.UnitRegistry()

    # Define snapshot base units and code units (if present)
    for group_name, prefix in (("Units", "snap"),
                               ("InternalCodeUnits", "code")):
        if group_name in snap:
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

    # Add some other units which might be useful
    unyt.define_unit("swift_mpc",  1.0e6*physical_constants_cgs["parsec"]*unyt.cm, registry=reg)
    unyt.define_unit("swift_msun", physical_constants_cgs["solar_mass"]*unyt.g, registry=reg)
    unyt.define_unit("newton_G", physical_constants_cgs["newton_G"]*unyt.cm**3/unyt.g/unyt.s**2, registry=reg)

    return reg


def soap_units_from_attributes(attrs, registry):
    """
    Create a SOAP style unyt.Unit object from dataset attributes

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
    try:
        cgs_conversion_from_attrs = float(attrs["Conversion factor to physical CGS (including cosmological corrections)"])
    except KeyError:
        cgs_conversion_from_attrs = float(attrs["Conversion factor to CGS (including cosmological corrections)"])
    cgs_conversion_from_unyt = float((1.0*unit).in_cgs().value)
    factor = cgs_conversion_from_attrs / cgs_conversion_from_unyt
    if np.isclose(factor, 1.0, atol=0.0, rtol=1.0e-9):
        return unit
    else:
        return unit*factor

    
def swiftsimio_units(group):
    """
    Given a HDF5 group containing swift unit information, return a
    swiftsimio units object.

    This is useful for SOAP output, where the unit information
    may be in a sub-group in the file.
    """
    
    buf = io.BytesIO()
    with h5py.File(buf, 'w') as tmpfile: 
        group.copy("Cosmology", tmpfile)
        group.copy("PhysicalConstants", tmpfile)
        group.copy("Units", tmpfile)
    return swiftsimio.reader.SWIFTUnits(buf)


def swiftsimio_cosmology(group):
    """
    Generate an astropy cosmology object via swiftsimio, given a HDF5
    group containing SWIFT metadata.
    """

    # Construct the SWIFTUnits object
    units = swiftsimio_units(group)

    # Extract cosmology
    cosmo = {}
    for name in group["Cosmology"].attrs:
        cosmo[name] = group["Cosmology"].attrs[name]
    
    return swiftsimio.conversions.swift_cosmology_to_astropy(cosmo, units)


def swiftsimio_units_from_attributes(attrs, units):
    """
    Given dataset attributes and a SWIFTUnits object, return
    swiftsimio style units and a flag to indicate if the dataset
    is comoving.
    """

    # Determine unyt unit for this quantity
    u = unyt.dimensionless
    for dim, unit in (("I", units.current),
                      ("L", units.length),
                      ("M", units.mass),
                      ("T", units.temperature),
                      ("t", units.time)):
        exponent = attrs["U_%s exponent" % dim][0]
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

    # Verify no h dependence, because this is not implemented
    h_scale_exponent = attrs["h-scale exponent"][0]
    if h_scale_exponent != 0:
        raise ValueError("Can't handle output with h dependent units!")
    
    # Check for comoving units
    a_scale_exponent = attrs["a-scale exponent"][0]
    length_exponent = attrs["U_L exponent"][0]
    if a_scale_exponent == 0.0:
        comoving = False
    elif a_scale_exponent == length_exponent:
        comoving = True
    else:
        raise ValueError("Can't determine if dataset is comoving!")
    
    # SOAP outputs can have units which are not just powers of the base units.
    # Note that the units here do not include powers of a, so need to use CGS factor
    # without cosmological corrections.
    cgs_conversion_from_attrs = float(attrs["Conversion factor to CGS (not including cosmological corrections)"])
    cgs_conversion_from_unyt = float((1.0*u).in_cgs().value)
    factor = cgs_conversion_from_attrs / cgs_conversion_from_unyt
    if not np.isclose(factor, 1.0, atol=0.0, rtol=1.0e-9):
        u = u*factor

    return u, comoving
        

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
    
    def __init__(self, obj, mode="swiftsimio"):
        """
        This handles opening a dataset. We need to read the dataset
        specific metadata attributes and store them.
        """
        super(SwiftDataset, self).__init__(obj)
        self.mode = mode
        
        self.attrs = {}
        for name, value in self.obj.attrs.items():
            self.attrs[name] = value
    
    def __getitem__(self, key):
        """
        This handles reading the actual data and returning a unyt quantity
        with suitable units.
        """
        data = self.obj[key]
        if self.mode == "soap":
            units =  soap_units_from_attributes(self.attrs, self.soap_registry)        
            return unyt.array.unyt_array(data, units, registry=self.soap_registry)
        elif self.mode == "swiftsimio":
            units, comoving = swiftsimio_units_from_attributes(self.attrs, self.swiftsimio_units)
            a_symbol = swiftsimio.objects.a
            a_value = self.expansion_factor
            a_exponent = self.attrs["a-scale exponent"][0]
            cosmo_factor = swiftsimio.objects.cosmo_factor(a_symbol**a_exponent, scale_factor=a_value)
            return swiftsimio.objects.cosmo_array(data, units=units, comoving=comoving, cosmo_factor=cosmo_factor)
        else:
            raise ValueError(f"Unrecognized mode parameter {self.mode}")
        

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
            result.soap_registry = self.soap_registry
            result.swiftsimio_units = self.swiftsimio_units
            result.mode = self.mode
            result.cosmology = self.cosmology
            result.expansion_factor = self.expansion_factor
        return result

    def get_unit(self, name):
        """
        Given a unit name, return a Unit object
        """
        return unyt.Unit(name, registry=self.soap_registry)


class SwiftFile(SwiftGroup):
    """
    This is a wrapper around the h5py.File object for a file
    containing arrays with Swift style metadata.

    All arguments are passed to the underlying h5py.File.
    """
    def __init__(self, mode, metadata_path, *args, **kwargs):

        self.mode = mode
        
        # Open the HDF5 file
        super(SwiftFile, self).__init__(h5py.File(*args, **kwargs))

        # Read SOAP style unit information
        self.soap_registry = soap_unit_registry_from_snapshot(self.obj[metadata_path])

        # Read swiftsimio units
        self.swiftsimio_units = swiftsimio_units(self.obj[metadata_path])
        
        # Read cosmology
        self.cosmology = swiftsimio_cosmology(self.obj[metadata_path])
        self.expansion_factor = self.obj[metadata_path]["Cosmology"].attrs["Scale-factor"][0]


class SwiftSnapshot(SwiftFile):
    """
    This is a wrapper around the h5py.File object for an
    open snapshot file.
    """
    def __init__(self, *args, mode="swiftsimio", **kwargs): 
        metadata_path="/"
        super(SwiftSnapshot, self).__init__(mode, metadata_path, *args, **kwargs)
