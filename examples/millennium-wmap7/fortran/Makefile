# Assume HDF5 wrapper has been built in "build" directory under source dir
HDF5_WRAPPER = ../../../fortran/hdf5_wrapper/build/

F90      = gfortran
F90FLAGS = -I$(HDF5_WRAPPER)/include/ -g
LINK     = -L$(HDF5_WRAPPER)/lib/ -Wl,-rpath=$(HDF5_WRAPPER)/lib/ -lhdfwrapper

.SUFFIXES:
.SUFFIXES: .f90 .o

.f90.o:
	$(F90) $(F90FLAGS) -c $< -o $*.o

read_mr7_snapshot_file:	read_mr7_snapshot_file.o
	$(F90) $(F90FLAGS) read_mr7_snapshot_file.o $(LINK) -o read_mr7_snapshot_file
clean:
	\rm -f *.o *.mod read_mr7_snapshot_file
