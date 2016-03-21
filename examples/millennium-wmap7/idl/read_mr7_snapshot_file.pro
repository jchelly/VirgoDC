
function read_dataset, file_id, name
;
; Read a hdf5 dataset from the specified file
;
  dset_id = h5d_open(file_id, name)
  data    = h5d_read(dset_id)
  h5d_close, dset_id
  
  return, data
end


function read_group_attribute, file_id, group_name, attr_name
;
; Read an attribute of a HDF5 group (e.g. Gadget snapshot header)
;
  group_id = h5g_open(file_id, group_name)
  attr_id  = h5a_open_name(group_id, attr_name)
  data = h5a_read(attr_id)
  h5a_close, attr_id
  h5g_close, group_id

  return, data
end


;
; Read a Millennium-WMAP7 snapshot file.
;
; Leaves particle positions, velocities and ids
; in variables pos, vel, ids.
;

; Name of the file to read
fname = "/gpfs/data/Millgas/data/dm/500/snapdir_061/500_dm_061.0.hdf5"

; Open file
file_id = h5f_open(fname)

; This shows how to read header entries
boxsize = read_group_attribute(file_id, "Header", "BoxSize")
print, "Box size is ", boxsize, " comoving Mpc/h"

numpart_thisfile = read_group_attribute(file_id, "Header", "NumPart_ThisFile")
print, "Number of particles in this file is ", numpart_thisfile[1]

; Read positions, velocities and IDs
pos = read_dataset(file_id, "PartType1/Coordinates")
vel = read_dataset(file_id, "PartType1/Velocities")
ids = read_dataset(file_id, "PartType1/ParticleIDs")

; Close the file
h5f_close, file_id

end

