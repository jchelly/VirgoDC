//
// Fortran callable L-Gadget read routines for reading subsections of
// files
//
// reapartcoords, readpartvel and readpartids read the coordinates, 
// velocities or IDs for the specified range of particles in the file.
//
// e.g. CALL readpartcoords(4,23,x,y,z)
// would read the coordinates of the 4th to the 23rd (inclusive) particles
// in the file (particle numbering starts at one here, although in the hash
// table the first particle is numbered zero)
//
// The hash table can be read with readhash()
//
// The file needs to have been opened first with open_gadget() in all cases.
//
// Compile with -DBYTESWAP to read millennium files on little endian machines
//

#include <stdio.h>
#include <stdlib.h>
FILE *fgad;

// Byte swapping routines
void byte_swap2(char *);
void byte_swap4(char *);
void byte_swap8(char *);

typedef struct
{
  int npart[6];
  double mass[6];
  double time;
  double redshift;
  int flag_sfr;
  int flag_feedback;
  int npartTotal[6];
  int flag_cooling;
  int num_files;
  double BoxSize;
  double Omega0;
  double OmegaLambda;
  double HubbleParam;
  int flag_stellarage;
  int flag_metals;
  int hashtabsize;
  char fill[84];
}
io_header;

// Abort and produce a message
void abortmessagepart(char *message)
{
  printf("Error in readparticles.c: \n");
  printf(message);
  abort();
}

//
// Open a file, read only, abort if an error occurs
//

void open_gadget_(char *name)
{
  char filename[256];
  int i;
  for (i=0; i<256; i++){
    if (name[i]==32) filename[i]=0;
    else 
      filename[i]=name[i];
  }
  fgad = fopen(filename,"r");
  if (fgad==0)abortmessagepart("Unable to open Gadget file");
}

// Close the file
void close_gadget_()
{
  fclose(fgad);
}

// Read specified particle coordinates
// Identifies particles by order in file
// Numbering starts from 1 for the first particle. 
void readpartcoords_(int *ifirst, int *ilast, float *x, float *y, float *z)
{
  long int offset;
  size_t bsize;
  float *block;
  int nread;
  int ipart;

  // Number of bytes to read
  bsize=(*ilast-*ifirst+1)*12;
  // Number of particles to read
  nread=*ilast-*ifirst+1;
  // Allocate temporary storage for particles
  block=malloc(bsize);
  // Position in file (in bytes) of the particles
  offset=(256+((*ifirst)*12));
  // Jump to correct position in file
  if(fseek(fgad, offset, SEEK_SET)!=0)
    abortmessagepart("fseek failed in readpartcoords() \n");
  // Read required amount of data into temporary block
  if(fread(block,12,(*ilast-*ifirst+1),fgad)!=nread)
    abortmessagepart("read block failed in readpartcoords() \n");
  // Store coords in arrays to be returned
  for(ipart=0;ipart<nread;ipart++)
    {
#ifdef BYTESWAP
      byte_swap4((char *)&(block[ipart*3]));
      byte_swap4((char *)&(block[ipart*3+1]));
      byte_swap4((char *)&(block[ipart*3+2]));
#endif
      x[ipart]=block[ipart*3];
      y[ipart]=block[ipart*3+1];
      z[ipart]=block[ipart*3+2];
    }
  free(block);
}

// Read specified particle velocities
// Identifies particles by order in file
// Numbering starts from 1 for the first particle. 
void readpartvel_(int *ifirst, int *ilast, float *vx, float *vy, float *vz)
{
  long int offset;
  size_t bsize;
  float *block;
  int nread;
  int ipart;
  int npfile;

  // Number of bytes to read
  bsize=(*ilast-*ifirst+1)*12;
  // Number of particles to read
  nread=*ilast-*ifirst+1;
  // Allocate temporary storage for particles
  block=malloc(bsize);
  // Find out how many particles are in this file
  if(fseek(fgad, 8, SEEK_SET)!=0)
    abortmessagepart("fseek failed in readpartvel (1) \n");
  if(fread(&npfile,sizeof(int),1,fgad)!=1)
    abortmessagepart("Unable to read particle number \n");
#ifdef BYTESWAP
  byte_swap4((char *) &npfile);
#endif
  // Position in file (in bytes) of the particles
  offset=264+(npfile*12)+(*ifirst*12);
  // Jump to correct position in file
  if(fseek(fgad, offset, SEEK_SET)!=0)
    abortmessagepart("fseek failed in readpartvel (2) \n");
  // Read required amount of data into temporary block
  if(fread(block,12,(*ilast-*ifirst+1),fgad)!=nread)
    abortmessagepart("read block failed in readpartcoords() \n");
  // Store coords in arrays to be returned
  for(ipart=0;ipart<nread;ipart++)
    {
#ifdef BYTESWAP
      byte_swap4((char *)&(block[ipart*3]));
      byte_swap4((char *)&(block[ipart*3+1]));
      byte_swap4((char *)&(block[ipart*3+2]));
#endif
      vx[ipart]=block[ipart*3];
      vy[ipart]=block[ipart*3+1];
      vz[ipart]=block[ipart*3+2];
    }
  free(block);
}

// Read particle IDs
void readpartids_(int *ifirst, int *ilast, long long *ids)
{
  long int offset;
  int nread;
  int npfile;
  int i;

  // Number of particles to read
  nread=*ilast-*ifirst+1;
  // Find out how many particles are in this file
  if(fseek(fgad, 8, SEEK_SET)!=0)
    abortmessagepart("fseek failed in readpartids (1) \n");
  if(fread(&npfile,4,1,fgad)!=1)
    abortmessagepart("Unable to read particle number \n");
#ifdef BYTESWAP
  byte_swap4((char *) &npfile);
#endif
  // Position in file (in bytes) of the particles
  offset=276+(npfile*24)+(*ifirst*8);
  // Jump to correct position in file
  if(fseek(fgad, offset, SEEK_SET)!=0)
    abortmessagepart("fseek failed in readpartids (2) \n");
  // Read required amount of data into ID array
  if(fread(ids,8,(*ilast-*ifirst+1),fgad)!=nread)
    abortmessagepart("read ids failed in readpartids \n");
#ifdef BYTESWAP
  for(i=0;i<nread;i++){
    byte_swap8((char *) &(ids[i]));
  }
#endif
}

// Read the hash table. Also returns first and last cells
// stored in this file
//
// Hash table entries call the first particle 0, not 1!
//
void readhash_(int *firstcell, int *lastcell, int *htable, int *npfile)
{
  long int offset;
  int hlen[2];
  int nhash;
  int i;
  // Find out how many particles are in this file
  if(fseek(fgad, 8, SEEK_SET)!=0)
    abortmessagepart("fseek failed in readhash (1) \n");
  if(fread(npfile,4,1,fgad)!=1)
    abortmessagepart("Unable to read particle number \n");
#ifdef BYTESWAP
  byte_swap4((char *) npfile);
#endif
  // Position in file (in bytes) of the hash table length info.
  offset=292+(*npfile*32);
  // Jump to correct position in file
  if(fseek(fgad, offset, SEEK_SET)!=0)
    abortmessagepart("fseek failed in readhash (2) \n");
  // Read hash table length
  if(fread(hlen,4,2,fgad)!=2)
    abortmessagepart("read ids failed in readhash \n");
#ifdef BYTESWAP
  byte_swap4((char *) &(hlen[0]));
  byte_swap4((char *) &(hlen[1]));
#endif
  *firstcell=hlen[0];
  *lastcell=hlen[1];
  nhash=hlen[1]-hlen[0]+1;
  // Read the table
  offset=offset+16;
  if(fseek(fgad, offset, SEEK_SET)!=0)
    abortmessagepart("fseek failed in readhash (3) \n");    
  if(fread(htable,4,nhash,fgad)!=nhash)
    abortmessagepart("Unable to read hash table \n");
#ifdef BYTESWAP
  for(i=0;i<nhash;i++){
    byte_swap4((char *)&(htable[i]));
  }
#endif
}

void byte_swap8(char *val)
{
char temp;


   temp = val[0];
   val[0] = val[7];
   val[7] = temp;

   temp = val[1];
   val[1] = val[6];
   val[6] = temp;

   temp = val[2];
   val[2] = val[5];
   val[5] = temp;

   temp = val[3];
   val[3] = val[4];
   val[4] = temp;
}

void byte_swap4(char *val)
{
char temp;

   temp = val[0];
   val[0] = val[3];
   val[3] = temp;

   temp = val[1];
   val[1] = val[2];
   val[2] = temp;
}

void byte_swap2(char *val)
{
char temp;

   temp = val[0];
   val[0] = val[1];
   val[1] = temp;
}
