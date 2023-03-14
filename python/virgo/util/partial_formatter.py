#!/bin/env python

import string

class FormatKey:
    def __init__(self, key):
        self.key = key

class PartialFormatter(string.Formatter):
    """
    Formatter which can leave unspecified values unchanged.

    If ignore_missing is True then any values not provided are left unchanged.
    E.g.

    > formatter = PartialFormatter(ignore_missing=True)
    > formatter.format("{a} {b} {c}", a=1, b=2)
    > '1 2 {c}'

    If ignore_missing is False then all values must be specified but they may
    be set to None if no value is to be substituted. E.g.

    > formatter = PartialFormatter()
    > formatter.format("{a} {b} {c}", a=1, b=2, c=None)
    > '1 2 {c}'

    In this case missing values will cause a KeyError exception.
    """
    def __init__(self, *args, ignore_missing=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_missing = ignore_missing

    def get_value(self, key, args, kwargs):
        if isinstance(key, int):
            # Using positional arguments, so just return value
            return args[key]
        elif (key not in kwargs) and self.ignore_missing:
            # Key not found, but we're leaving missing keys unchanged
            return FormatKey(key) 
        elif (key in kwargs) and (kwargs[key] is None):
            # Key found but set to None, so leave it unchanged
            return FormatKey(key) 
        else:
            # Otherwise return the value
            return kwargs[key]

    def format_field(self, value, format_spec):
        if isinstance(value, FormatKey):
            # We're not substituting anything in here
            if len(format_spec) > 0:
                format_spec=f":{format_spec}"
            return "{"+value.key+format_spec+"}"
        else:
            # Substitute value as normal
            return format(value, format_spec)
