; prints no particles in group with igrp-th largest m200 using group
; catalogues, and prints m200 of from

bmilli=0
num = 63
exts='000'
exts=exts+strcompress(string(num),/remove_all)
exts=strmid(exts,strlen(exts)-3,3)

ofname= "group_m200_"+ exts +".all" 
ofname=strcompress(ofname,/remove_all)
openr, 1, ofname
totnsubs = 0L
readu, 1, totnsubs

M200 = fltarr(totnsubs)
Filenr = lonarr(totnsubs)
idsort = lonarr(totnsubs)
FirstSubOfHalo = lonarr(totnsubs)

readu, 1, m200
readu, 1, filenr
readu, 1, idsort
readu, 1, firstsubofhalo
close, 1

igrp = 1000

Subfile = filenr[igrp]
print, "SubFile= ",Subfile, string(subfile)

readgroups, Num, Subfile, Base, Snapbase, $
                Nfiles, $
                Ngroups, $
                Nids, $
                TotNgroups, $
                Grouplen, $
                GroupOffset, bmilli=bmilli

exts='000'
exts=exts+strcompress(string(num),/remove_all)
exts=strmid(exts,strlen(exts)-3,3)
Outputdir  = Base + "/snapdir_"+exts+"/"

print, "From group_m200_" + exts+ ".all; ", m200[igrp]

snap_header, Num, 0, outputdir, snapbase, $
             npart, massarr, time, redshift, flag_sfr, flag_feedback, $
             npartall, flag_cooling, Nsubfiles, BoxSize, Hubble,RhoCrit,RhoBack

PartMass= massarr(1)


print, "From group catlogue: ", GroupLen[idsort(igrp)], GroupLen(idsort(igrp))*PartMass


readsubhalos, Num, SubFile, Base, snapbase,$
                     Ngroups, $
                     Nids, $
                     TotNgroups, $
                     Nfiles, $
                     TotNsubs, $
                     Nsubs, $
                     NsubPerhalo, $
                     FirstSubofHalo, $
                     Sublen, $
                     SubOffset, $
                     SubParenthalo, $
                     Halo_M_Mean200, $
                     Halo_R_Mean200, $
                     Halo_M_Crit200, $
                     Halo_R_Crit200, $
                     Halo_M_TopHat200, $
                     Halo_R_TopHat200, $
                     SubPos, $
                     SubVel, $
                     SubVelDisp, $
                     SubVmax, $
                     SubSpin, $
                     SubmostBoundID, $
                     SubHalfMass, bmilli=bmilli

print, "From subhalo catlogue: ", Halo_M_Crit200[idsort(igrp)],  Halo_M_Crit200[idsort(igrp)]*1e10
print, firstsubofhalo[idsort[igrp]], subparenthalo[firstsubofhalo[idsort[igrp]]], idsort[igrp]


end
