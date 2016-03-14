
Base= "/virgo/data/MilliMillennium/"
SnapBase="snap_millennium/"

Num = 8 


exts='000'
exts=exts+strcompress(string(Num),/remove_all)
exts=strmid(exts,strlen(exts)-3,3)

f= Base + "/snapdir_"+exts+"/"+ snapbase+"_"+exts +".0"
f= Strcompress(f, /remove_all)

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
close,1


print, npartall
print,time,redshift
print, boxsize


count=0L
for subfile=0, Nsubfiles-1 do begin

    f=base + "/snapdir_"+exts+"/"+ snapbase+"_"+exts +"." + string(subfile)
    f=strcompress(f,/remove_all)
   
    openr,1,f,/f77_unformatted
    readu,1,npart,massarr,time,redshift,flag_sfr,flag_feedback,npartall
    print,npart,massarr
    print
    N=npart(1)
    pos1 = fltarr(3,N)
    readu,1,pos1
    close,1

    if subfile eq 0 then begin
       Pos= fltarr(3,npartall(1))
    endif

    Pos(*, count:count+N-1)= Pos1(*,*)
    count=count+N
endfor
    
ind=where(pos(2,*) lt BoxSize/50)


window, xsize=800, ysize=800, retain=2 

plot, pos(0,ind), pos(1,ind), psym=3

end

