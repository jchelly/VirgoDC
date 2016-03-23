#include "FC.h"

/* Return 0 for big endian, 1 for little endian */
void FC_GLOBAL_(byte_order, BYTE_ORDER)(int *i)
{                                                   
  int   one = 1;
  char* endptr = (char *) &one;
  *i = (*endptr);
  return;
} 
