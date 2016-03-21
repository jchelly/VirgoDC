
function read_dataset, file_id, name
;
; Read a hdf5 dataset from the specified file
;
  dset_id = h5d_open(file_id, name)
  data    = h5d_read(dset_id)
  h5d_close, dset_id
  
  return, data
end

;
; Read a Millennium-WMAP7 snapshot file.
;

; Name of the file to read
fname = "/gpfs/data/Millgas/data/dm/500/snapdir_061/500_dm_061.0.hdf5"

; Open file
file_id = h5f_open(fname)

; Read positions, velocities and IDs
pos = read_dataset(file_id, "PartType1/Coordinates")
vel = read_dataset(file_id, "PartType1/Velocities")
ids = read_dataset(file_id, "PartType1/ParticleIDs")

; Close the file
h5f_close, file_id

end

