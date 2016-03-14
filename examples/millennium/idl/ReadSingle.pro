;
; IDL code to read a single Millennium snapshot file
;

Base= "/virgo/data/Millennium/"
SnapBase="snap_millennium"

Num = 26        ; number of snapshot to read
Subfile= 147    ; which file within the snapshot


exts='000'
exts=exts+strcompress(string(Num),/remove_all)
exts=strmid(exts,strlen(exts)-3,3)

f=base + "/snapdir_"+exts+"/"+ snapbase+"_"+exts +"." + string(subfile)
f=strcompress(f,/remove_all)

npart=lonarr(6)		
massarr=dblarr(6)
time=0.0D
redshift=0.0D
flag_sfr=0L
flag_feedback=0L
npartall=lonarr(6)	
flag_cooling= 0L
Nsubfiles = 0L
BoxSize = 0.0D

openr,1,f,/f77_unformatted
readu,1, npart, massarr, time, redshift, flag_sfr, flag_feedback, $
         npartall, flag_cooling, Nsubfiles, BoxSize 
N= npart(1)
pos = fltarr(3,N)
vel = fltarr(3,N)
id =  lon64arr(N)
readu,1,pos
readu,1,vel
readu,1,id
close,1


end

















 
