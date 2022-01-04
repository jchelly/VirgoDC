#!/usr/bin/env python

from setuptools import setup

setup(name='Virgo data formats',
      version='0.1',
      description='Code for reading various Virgo simulation outputs',
      author='John Helly',
      author_email='j.c.helly@durham.ac.uk',
      packages=['virgo','virgo.formats','virgo.util','virgo.sims','virgo.database','virgo.mpi'],
     )

