F90      = gfortran
F90FLAGS = -g
LINK     =

.SUFFIXES:
.SUFFIXES: .f90 .o

.f90.o:
	$(F90) $(F90FLAGS) -c $< -o $*.o

read_m2_snapshot_file:	read_m2_snapshot_file.o
	$(F90) $(F90FLAGS) read_m2_snapshot_file.o $(LINK) -o read_m2_snapshot_file

clean:
	\rm -f *.o *.mod read_m2_snapshot_file
