HDF5     = $(HDF5_HOME)
CC       = gcc
CFLAGS   = -O2 -g -I$(HDF5_HOME)/include/
LDFLAGS  = -L$(HDF5_HOME)/lib/ -Wl,-rpath=$(HDF5_HOME)/lib/

read_mr7_snapshot_file:	read_mr7_snapshot_file.o
	$(CC) $(CFLAGS) read_mr7_snapshot_file.o -o read_mr7_snapshot_file $(LDFLAGS) -lhdf5 -lhdf5_hl

clean:
	\rm -rf *.o
