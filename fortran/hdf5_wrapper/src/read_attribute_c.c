#include "hdf5.h"
#include "FC.h"

// Routine to write an HDF5 dataset from Fortran - this should allow writing
// of data types not defined in the HDF5 Fortran interface, although it does
// circumvent some Fortran error checking...

void FC_GLOBAL_(read_hdf5_attribute, READ_HDF5_ATTRIBUTE)(hid_t *pattr_id, hid_t *pmem_type_id, 
							  void *buf, herr_t *err)

{
  *err = H5Aread(*pattr_id,*pmem_type_id,buf);
}


