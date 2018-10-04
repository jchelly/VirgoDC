#!/bin/env python

from mpi4py import MPI
import traceback
import sys

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

def mpi_errchk(func):
    """
    Decorator to call a function and call MPI_Abort if it
    raises an exception. Kill signals (e.g. from the batch 
    system) show up as KeyboardInterrupts so we catch those 
    too.
    """
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (Exception, KeyboardInterrupt) as e:
            sys.stderr.write("\n\n*** EXCEPTION ***\n"+str(e)+" on rank "+str(comm_rank)+"\n\n")
            traceback.print_exc(file=sys.stdout)
            sys.stderr.write("\n\n")
            sys.stderr.flush()
            comm.Abort(1)
    return func_wrapper



