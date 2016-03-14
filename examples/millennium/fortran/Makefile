## Compiler flags for gfortran on virgo analysis node
#
F90C = gfortran
FC   = gfortran
CC   = gcc
F90FLAGS = -g -O2
FFLAGS =   -g -O2
CFLAGS =   -g -O2

.SUFFIXES:
.SUFFIXES: .o .f .f90 .c

.f.o:
	$(FC) -c $< -o $*.o $(FFLAGS)

.f90.o:
	$(F90C) -c $< -o $*.o $(F90FLAGS)

.c.o:
	$(CC) -c $< -o $*.o $(CFLAGS)


all: readregionexample readgroupsexample readfileexample readsubhalosexample readfofgroupsexample

readregionexample: file_path.o peano_keys.o readregion.o readregionexample.o readparticles.o Makefile
	$(F90C) file_path.o  peano_keys.o readregion.o readregionexample.o readparticles.o $(F90FLAGS) -o readregionexample

readgroupsexample: file_path.o sort.o peano_keys.o readgroups.o readregion.o readgroupsexample.o readparticles.o Makefile
	$(F90C)  file_path.o sort.o peano_keys.o readgroups.o readregion.o readgroupsexample.o readparticles.o $(F90FLAGS) -o readgroupsexample

readsubhalosexample: file_path.o  readgroups.o readsubhalosexample.o Makefile
	$(F90C) file_path.o  readgroups.o readsubhalosexample.o -o readsubhalosexample $(F90FLAGS)

readfileexample: file_path.o  readfile.o readfileexample.o
	$(F90C) file_path.o  readfile.o readfileexample.o $(F90FLAGS) -o readfileexample

readfofgroupsexample: sort.o file_path.o  readfile.o peano_keys.o readgroups.o readregion.o readfofgroupsexample.o readparticles.o Makefile
	$(F90C)  file_path.o readfile.o  peano_keys.o readgroups.o readregion.o readfofgroupsexample.o readparticles.o sort.o $(F90FLAGS) -o readfofgroupsexample

clean:
	rm -f *.o *.mod readregionexample readgroupsexample readfileexample readsubhalosexample readfofgroupsexample
