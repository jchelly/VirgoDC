

Base= "/ptmp/vrs/milliMill/"       ;;;;; path where stuff is stored
Num=     63                        ;;;;; number of snapshot
Subfile= 0                         ;;;;; subfile
 

exts='000'
exts=exts+strcompress(string(num),/remove_all)
exts=strcompress(strmid(exts,strlen(exts)-3,3),/remove_all)

fsubgrp=   base+"/postproc_"+exts+"/sub_tab_"+exts+"."+string(Subfile)    
fsubgrp=  strcompress(fsubgrp, /remove_all)

fgrp=   base+"/snapdir_"+exts+"/group_tab_"+exts+"."+string(Subfile)    
fgrp=  strcompress(fgrp, /remove_all)


openr,1,fgrp      
Ngroups=0L
readu,1,Ngroups
print,"Ngroups= ", Ngroups
Nids=0L
readu,1,Nids
TotNgroups=0L
readu,1,TotNgroups
print,"TotNgroups= ", TotNgroups
Nfiles=0L
readu,1,Nfiles
GroupLen = lonarr(Ngroups)
readu,1, GroupLen
close,1


openr,1,fsubgrp      
Ngroups=0L
readu,1,Ngroups
print,"Ngroups= ", Ngroups
Nids=0L
readu,1,Nids
TotNgroups=0L
readu,1,TotNgroups
print,"TotNgroups= ", TotNgroups
Nfiles=0L
readu,1,Nfiles

Nsubs=0L
readu,1,Nsubs
print,"Nsubs= ", Nsubs

NsubPerHalo = lonarr(Ngroups)
FirstSubOfHalo = lonarr(Ngroups)
SubLen=lonarr(Nsubs)
SubOffset=lonarr(Nsubs)
SubParentHalo= lonarr(Nsubs)

readu,1, NsubPerHalo , FirstSubOfHalo, SubLen, SubOffset, SubParentHalo


Halo_M_Mean200 = fltarr(Ngroups)
Halo_R_Mean200 = fltarr(Ngroups)
Halo_M_Crit200 = fltarr(Ngroups)
Halo_R_Crit200 = fltarr(Ngroups)
Halo_M_TopHat200 = fltarr(Ngroups)
Halo_R_TopHat200 = fltarr(Ngroups)

readu,1, Halo_M_Mean200, Halo_R_Mean200 
readu,1, Halo_M_Crit200, Halo_R_Crit200 
readu,1, Halo_M_TopHat200, Halo_R_TopHat200

SubPos = fltarr(3, Nsubs)
SubVel = fltarr(3, Nsubs)
SubVelDisp = fltarr(Nsubs)
SubVmax = fltarr(Nsubs)
SubSpin = fltarr(3, Nsubs)
SubMostBoundID = lon64arr(Nsubs)
SubHalfMass = fltarr(Nsubs)

readu,1, SubPos, SubVel, SubVelDisp, SubVmax, SubSpin, SubMostBoundID, SubHalfMass

close,1



end
