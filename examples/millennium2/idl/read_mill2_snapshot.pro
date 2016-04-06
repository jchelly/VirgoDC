;
; Read a Millennium2 snapshot file.
;
; Leaves particle positions, velocities and ids
; in variables pos, vel, ids.
;

; Name of the file to read
fname = "/data/millennium2/millennium-2/snapdir_067/snap_newMillen_067.0"

; Struct to contain header data
header = CREATE_STRUCT("npart",lonarr(6),"mass",dblarr(6),"time",DOUBLE(-1.0),"redshift",DOUBLE(-1),"flag_sfr",0L,"flag_feedback",0L,"npartTotal",ulonarr(6),"flag_cooling",0L,"num_files",0L,"BoxSize",DOUBLE(-1),"Omega0",DOUBLE(0.0),"OmegaLambda",DOUBLE(0.0),"HubbleParam",DOUBLE(0.0),"flag_stellarage", 0L, "flag_metals", 0L, "npartTotalHighWord", ulonarr(6), "unused", bytarr(6))

; Open file and read header
openr, iunit, fname, /GET_LUN, /F77_UNFORMATTED
readu, iunit, header

; Read positions
pos = fltarr(3, header.npart[1])
readu, iunit, pos

; Read velocities
vel = fltarr(3, header.npart[1])
readu, iunit, vel

; Read particle IDs
ids = lon64arr(header.npart[1])
readu, iunit, ids

free_lun, iunit
