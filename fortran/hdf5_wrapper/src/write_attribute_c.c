#include "hdf5.h"
#include "FC.h"

void FC_GLOBAL_(write_hdf5_attribute, WRITE_HDF5_ATTRIBUTE)(hid_t *pattr_id, hid_t *pmem_type_id, 
							    void *buf, herr_t *err)
{
  *err = H5Awrite(*pattr_id,*pmem_type_id,buf);
}


