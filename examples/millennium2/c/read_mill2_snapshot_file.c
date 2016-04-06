#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "hdf5.h"
#include "hdf5_hl.h" 

struct io_header
{
  int npart[6];			
  double mass[6];		
  double time;			
  double redshift;		
  int flag_sfr;			
  int flag_feedback;		
  unsigned int npartTotal[6];	
  int flag_cooling;		
  int num_files;		
  double BoxSize;		
  double Omega0;		
  double OmegaLambda;		
  double HubbleParam;		
  int flag_stellarage;		
  int flag_metals;		
  unsigned int npartTotalHighWord[6];	
  char unused[64];
};


/*
  Read the specified snapshot file from the Millennium-2 run.
  Writes positions, velocities and particle IDs to stdout.
*/
int main(int argc, char *argv[])
{
  struct io_header header;
  FILE *fd;
  int irec;
  float *pos;
  float *vel;
  long long *ids;

  /* Take file name from first command line argument */
  if ( argc != 2 )
    {
      printf("Usage: read_ill2_snapshot_file <filename>\n");
      exit(1);
    }
  char fname[500];
  strncpy(fname, argv[1], 500);

  /* Open the file and read header */
  if(!(fd = fopen(fname, "r")))
    {
      printf("Unable to open file: %s\n", fname);
      exit(1);
    }
  fread(&irec, sizeof(int),   1,    fd);
  if(irec != 256)
    {
      printf("Start of header record has wrong length!");
      exit(2);
    }
  fread(&header, sizeof(struct io_header), 1, fd);
  fread(&irec, sizeof(int),   1,    fd);
  if(irec != 256)
    {
      printf("End of header record has wrong length!");
      exit(2);
    }

  /* Find number of particles */
  int np = header.npart[1];
  printf("Number of particles in this file = %d\n", np);

  /* Read positions */
  pos = malloc(sizeof(float)*3*np);
  fread(&irec, sizeof(int),   1,    fd);
  fread(pos,   sizeof(float), 3*np, fd);
  fread(&irec, sizeof(int),   1,    fd);

  /* Read velocities */
  vel = malloc(sizeof(float)*3*np);
  fread(&irec, sizeof(int),   1,    fd);
  fread(vel,   sizeof(float), 3*np, fd);
  fread(&irec, sizeof(int),   1,    fd);

  /* Read particle IDs*/
  ids = malloc(sizeof(long long)*np);
  fread(&irec, sizeof(int),       1,  fd);
  fread(ids,   sizeof(long long), np, fd);
  fread(&irec, sizeof(int),       1,  fd);

  /* Close the file */
  fclose(fd);

  /* Write particles to stdout */
  printf("# x, y, z, vx, vy, vz, id\n");
  int i;
  for(i=0;i<np;i+=1)
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
