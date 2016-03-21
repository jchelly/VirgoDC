
BaseDir =   "/virgo/data/Millennium/"    ; The output-directory of the simulation
SnapBase=   "snapshot"                ; The basename of the snapshot files

Num = 63                         ; The number of the snapshot we look at

;;; The object-file of the compile C-library for accessing the group catalogue

ObjectFile = "./idlgrouplib.so"

Num=long(Num)
exts='000'
exts=exts+strcompress(string(num),/remove_all)
exts=strmid(exts,strlen(exts)-3,3)
Outputdir  = Basedir + "/snapdir_"+exts+"/"



;;;;;;;;;; First, we get the minimum size of stored groups

GroupMinSize = CALL_EXTERNAL(ObjectFile, $
                       'get_minimum_group_len', /UL_VALUE, $
                        OutputDir, $
                        Num)

print, "Minimum size of groups in the catalogue: ", GroupMinSize


;;;;;;;;;; Now we load a short histogram with the counts below the minimum size

CountBelowMinimum = dblarr(GroupMinSize)

CountIds = 0.0D

GroupMinSize = CALL_EXTERNAL(ObjectFile, $
                       'get_groupcount_below_minimum_len', /UL_VALUE, $
                        OutputDir, $
                        Num, $
                        CountBelowMinimum, $
                        CountIds)


;;;;;;;;;; Now we get the number of groups

Ngroups = CALL_EXTERNAL(ObjectFile, $
                       'get_total_number_of_groups', /UL_VALUE, $
                        OutputDir, $
                        Num)

print, "Number of groups in the catalogue: ", Ngroups
print, "Number of particles in groups: ",     CountIds

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


;;;;;;;;;; Let us plot the cumulative mass function down to 1-particle "groups"

Count= lindgen(Ngroups) + 1

plot, GroupLen, Count, /xlog, /ylog , xrange = [1, max(GroupLen)]

CumulativeBelow = dblarr(GroupMinSize)

CumulativeBelow(GroupMinSize-1) = CountBelowMinimum(GroupMinSize-1) + Ngroups

i = GroupMinSize-2
repeat begin
  CumulativeBelow(i) = CumulativeBelow(i+1) + CountBelowMinimum(i)
  i = i - 1
endrep until i eq 0 


SizeBelow = lindgen(GroupMinSize)

oplot, SizeBelow(1:*), CumulativeBelow(1:*), color=255

print, total(GroupLen,/double) + total( CountBelowMinimum * SizeBelow ,/double)


end
