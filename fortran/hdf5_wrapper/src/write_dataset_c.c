#include "hdf5.h"
#include "FC.h"

// Routine to write an HDF5 dataset from Fortran - this should allow writing
// of data types not defined in the HDF5 Fortran interface, although it does
// circumvent some Fortran error checking...

void FC_GLOBAL_(write_hdf5_dataset, WRITE_HDF5_DATASET)(hid_t *pdset_id, hid_t *pmem_type_id,
						       hid_t *dspace_id, hid_t *memspace_id,
						       void *buf, herr_t *err)
{
  *err = H5Dwrite(*pdset_id,*pmem_type_id,*memspace_id,*dspace_id,H5P_DEFAULT,
		  buf);
}


