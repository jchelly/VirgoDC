# Assume HDF5 wrapper has been built in source dir
HDF5_WRAPPER = ../../../fortran/hdf5_wrapper/

F90      = gcc
F90FLAGS = -I$(HDF5_WRAPPER)/include/ -g
LINK     = -L$(HDF5_WRAPPER)/lib/ -Wl,-rpath=$(HDF5_WRAPPER)/lib/ -lhdfwrapper

.SUFFIXES:
.SUFFIXES: .f90 .o

.f90.o:
	$(F90) $(F90FLAGS) -c $< -o $*.o

read_hdf5_snapshot:	read_hdf5_snapshot.o
	$(F90) $(F90FLAGS) read_hdf5_snapshot.o $(LINK) -o read_hdf5_snapshot
clean:
	\rm -f *.o *.mod read_hdf5_snapshot
