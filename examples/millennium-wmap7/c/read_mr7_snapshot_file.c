#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "hdf5.h"
#include "hdf5_hl.h" 

/*
  Read the specified snapshot file from the Millennium-WMAP7 (MR7) run
  using the hdf5 lite (H5LT) interface.
*/
int main(int argc, char *argv[])
{
  /* Take file name from first command line argument */
  if ( argc != 2 )
    {
      printf("Usage: read_mr7_snapshot_file <filename>\n");
      exit(1);
    }
  char fname[500];
  strncpy(fname, argv[1], 500);

  /* Open the file */
  hid_t file_id  = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  if(file_id < 0)
    {
      printf("Unable to open file: %s\n", fname);
      abort();
    }

  /* Read number of particles in this file */
  unsigned int NumPart_ThisFile[6];
  if(H5LTget_attribute_uint(file_id, "Header", "NumPart_ThisFile", NumPart_ThisFile) < 0)
    {
      printf("Unable to read number of particles from file %s.\n", fname);
      exit(1);
    }
  printf("Number of particles to read = %d\n", NumPart_ThisFile[1]);

  /* Can read header entries with H5LTget_attribute */
  double BoxSize;
  if(H5LTget_attribute_double(file_id, "Header", "BoxSize", &BoxSize) < 0)
    {
      printf("Unable to read box size from file %s.\n", fname);
      exit(1);
    }
  printf("Box size = %f\n", BoxSize);

  /* Read coordinates (comoving Mpc/h) */
  float *pos = malloc(3*NumPart_ThisFile[1]*sizeof(float));
  if(H5LTread_dataset(file_id, "PartType1/Coordinates", H5T_NATIVE_FLOAT, pos) < 0)
    {
      printf("Unable to read number of particles from file %s.\n", fname);
      exit(1);
    }

  /* Read velocities (multiply these by sqrt(a) to get peculiar velocity in km/sec) */
  float *vel = malloc(3*NumPart_ThisFile[1]*sizeof(float));
  if(H5LTread_dataset(file_id, "PartType1/Velocities", H5T_NATIVE_FLOAT, vel) < 0)
    {
      printf("Unable to read number of particles from file %s.\n", fname);
      exit(1);
    }

  /* Read particle IDs */
  unsigned long long *ids = malloc(NumPart_ThisFile[1]*sizeof(unsigned long long));
  if(H5LTread_dataset(file_id, "PartType1/ParticleIDs", H5T_NATIVE_ULLONG, ids) < 0)
    {
      printf("Unable to read number of particles from file %s.\n", fname);
      exit(1);
    }

  /* Close the hdf5 file */
  H5Fclose(file_id);

  /* Write particles to stdout */
  printf("# x, y, z, vx, vy, vz, id\n");
  int i;
  for(i=0;i<NumPart_ThisFile[1];i+=1)
    {
      printf("%14.6e, %14.6e, %14.6e, %14.6e, %14.6e, %14.6e, %llu\n",
	     pos[3*i+0], pos[3*i+1], pos[3*i+2],
	     vel[3*i+0], vel[3*i+1], vel[3*i+2],
	     ids[i]);
    }

  free(pos);
  free(vel);
  free(ids);

  return 0;
}
