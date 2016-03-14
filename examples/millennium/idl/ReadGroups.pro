
Base= "/ptmp/vrs/Millennium/"

Num = 63  ; number of snapshot files

Subfile = 5 ; number of subfile within snapshot

exts='000'
exts=exts+strcompress(string(Num),/remove_all)
exts=strmid(exts,strlen(exts)-3,3)


f= Base + "/snapdir_"+exts+"/group_tab_"+exts +"."+string(subfile)
f= Strcompress(f, /remove_all)

openr, 1, f

Ngroups=0L
Nids=0L
TotNgroups=0L
NTask=0L

readu, 1, Ngroups, Nids, TotNgroups, NTask

GroupLen= lonarr(NumLocal)
GroupOffset= lonarr(NumLocal) 

readu, 1, GroupLen
readu, 1, GroupOffset

MeangroupLen= 0L
readu, 1, MeangroupLen

CountBelowMinLen= lonarr(MeangroupLen)
readu, 1, CountBelowMinLen

close,1

end
