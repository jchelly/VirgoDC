CC       = gcc
CFLAGS   = -O2 -g

read_mr7_snapshot_file:	read_mr7_snapshot_file.o
	$(CC) $(CFLAGS) read_mr7_snapshot_file.o -o read_mr7_snapshot_file -lhdf5 -lhdf5_hl

clean:
	\rm -rf *.o
