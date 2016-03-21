
BaseDir =   "/virgo/data/Millennium/"    ; The output-directory of the simulation
SnapBase=   "snap_millennium"                   ; The basename of the snapshot files

Num = 63                          ; The number of the snapshot we look at

;;; The object-file of the compile C-library for accessing the group catalogue

ObjectFile = "./idlgrouplib.so"

Num=long(Num)
exts='000'
exts=exts+strcompress(string(num),/remove_all)
exts=strmid(exts,strlen(exts)-3,3)
Outputdir  = Basedir + "/snapdir_"+exts+"/"



;;;;;;;;;; First, we get the number of groups

Ngroups = CALL_EXTERNAL(ObjectFile, $
                       'get_total_number_of_groups', /UL_VALUE, $
                        OutputDir, $
                        Num)

print, "Number of groups in the catalogue: ", Ngroups

;;;;;;;;;; Now we load the group catalogue


GroupLen = lonarr(Ngroups)
GroupFileNr = lonarr(Ngroups)
GroupNr = lonarr(Ngroups)

Ngroups = CALL_EXTERNAL(ObjectFile, $
                       'get_group_catalogue', /UL_VALUE, $
                        OutputDir, $
                        Num, $
                        GroupLen, $
                        GroupFileNr, $
                        GroupNr)

;;; Now let's plot a simple cumulative mass function


Count= lindgen(Ngroups) + 1

plot, GroupLen, Count, /xlog, /ylog

end
