
Num = 63
FileNr = 0

Base = "/afs/mpa/sim/milliMill/"


exts= '000'
exts=exts+strcompress(string(num),/remove_all)

exts=strmid(exts,strlen(exts)-3,3)


fname= base+"/treedata/treeaux_"+ exts + ". " + string(FileNr)
fname=strcompress(fname,/remove_all)

openr,1, fname


TotNHalos = 0L
TotUniqueIDs = 0L
TotTrees = 0L
TotSnaps = 0L

readu,1,TotNhalos,TotUniqueIDs,TotTrees,TotSnaps


print, "TotNhalos=    ", TotNhalos
print, "TotUniqueIDs= ", TotUniqueIDs
print, "TotTrees=     ", TotTrees
print, "TotSnaps=     ", TotSnaps


CountID_Snap = lonarr(TotSnaps)
OffsetID_Snap = lonarr(TotSnaps)

CountID_SnapTree = lonarr(TotTrees, TotSnaps)
OffsetID_SnapTree = lonarr(TotTrees, TotSnaps)

CountID_Halo = lonarr(TotNhalos)
OffsetID_Halo = lonarr(TotNhalos)

readu,1, CountID_Snap, OffsetID_Snap

readu,1, CountID_SnapTree, OffsetID_SnapTree 

readu,1, CountID_Halo, OffsetID_Halo


IDs = lon64arr(TotUniqueIDs)
readu,1,IDs

Pos = fltarr(3, TotUniqueIDs)
readu,1,Pos

close,1

end

