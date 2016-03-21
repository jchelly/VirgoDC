;goto,jump

BaseDir =   "/virgo/data/Millennium/"    
SnapBase=   "snap_millennium"                   ; The basename of the snapshot files



Num = 39                     ; The number of the snapshot we look at

;;; The object-file of the compile C-library for accessing the group catalogue

ObjectFile = "./idlgrouplib.so"


Num=long(Num)
exts='000'
exts=exts+strcompress(string(num),/remove_all)
exts=strmid(exts,strlen(exts)-3,3)
Outputdir  = Basedir + "/snapdir_"+exts+"/"



fname = BaseDir + "/snapdir_"+exts+"/"+ snapbase+"_"+exts +"." + string(0)
fname=strcompress(fname,/remove_all)
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
openr,1,fname,/f77_unformatted
readu,1, npart, massarr, time, redshift, flag_sfr, flag_feedback, $
         npartall, flag_cooling, Nsubfiles, BoxSize 
close,1

PartMass= massarr(1)

print, "Time= ", Time


Hubble = 100.0 * sqrt(0.25/Time^3 + 0.75)

RhoCrit= 3.0 * Hubble^2  /(8*!PI * 43.0071)

RhoBack= 0.25*3.0 * 100.0^2  /(8*!PI * 43.0071) / time^3






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


;;;;;;;;;; Now we get the size of the hashtable

NFiles=0L

HashTabSize = CALL_EXTERNAL(ObjectFile, $
                       'get_hash_table_size', /UL_VALUE, $
                        OutputDir, $
                        Num, $
                        SnapBase, $
                        NFiles)




;;;;;;;;; Now we load the hash-table

HashTable= lonarr(HashTabSize)
FileTable= lonarr(HashTabSize)
LastHashCell = lonarr(NFiles)
NInFiles = lonarr(NFiles)

result = CALL_EXTERNAL(ObjectFile, $
                       'get_hash_table', /UL_VALUE, $
                        OutputDir, $
                        Num, $
                        SnapBase, $
                        HashTable, $
                        FileTable, $
                        LastHashCell, $
                        NInFiles)




jump:

window, xsize=1100, ysize=1100

!P.multi=[0,3,3]


start=  100

for GGN=start, start+8 do begin


N= long(GGN) ; group number

finr=  GroupFileNr(N) ; determines in which file this group is stored
grnr = GroupNr(N)     ; gives the group number within this file
Len = GroupLen(N)     ; gives the group length


Pos = Fltarr(3,Len)
Sx=0.0
Sy=0.0
Sz=0.0

result = CALL_EXTERNAL(ObjectFile, $
                       'get_group_coordinates', /UL_VALUE, $
                        OutputDir, $
                        Num, $
                        SnapBase, $
                        HashTable, $
                        FileTable, $
                        HashTabSize, $
                        LastHashCell, $
                        NInFiles, $ 
                        grnr, $
                        finr, $
                        Len, $
                        Pos, $
                        Sx, Sy, Sz)


;   Plot, Pos(0,*), Pos(1,*), psym=3



print, "GroupNr=", N, " length=", Len, "   center of mass=", sx, sy, sz



  x= Pos(0,*)
  y= Pos(1,*)
  z= Pos(2,*)

xxc =0.0
yyc =0.0
zzc =0.0


  r=sqrt(x^2 + y^2 + z^2)
  rmax = max(r)

  repeat begin
      r=sqrt(x^2 + y^2 + z^2)
      ind = where(r lt rmax)

      if n_elements(ind) ge 25 then begin
          xc= total(x(ind), /double)/n_elements(ind)
          yc= total(y(ind), /double)/n_elements(ind)
          zc= total(z(ind), /double)/n_elements(ind)

          x= x -xc
          y= y -yc
          z= z -zc

          xxc = xxc + xc
          yyc = yyc + yc
          zzc = zzc + zc


;          print, n_elements(ind), xxc, yyc, zzc
      endif

      rmax = 0.9 * rmax
  endrep until n_elements(ind) lt 25



Cx= float(Sx+xxc)
Cy= float(Sy+yyc)
Cz= float(Sz+zzc)

Rad = float(max(r))

Ncount = CALL_EXTERNAL(ObjectFile, $
                       'get_spherical_region_count', /UL_VALUE, $
                        OutputDir, $
                        Num, $
                        SnapBase, $
                        HashTable, $
                        FileTable, $
                        HashTabSize, $
                        LastHashCell, $
                        NInFiles, $ 
                        Cx, Cy, Cz, Rad)


print, "Ncount =", Ncount


PP = fltarr(3,Ncount)


Ncount = CALL_EXTERNAL(ObjectFile, $
                       'get_spherical_region_coordinates', /UL_VALUE, $
                        OutputDir, $
                        Num, $
                        SnapBase, $
                        HashTable, $
                        FileTable, $
                        HashTabSize, $
                        LastHashCell, $
                        NInFiles, $ 
                        Cx, Cy, Cz, Rad, PP)



r=sqrt(PP(0,*)^2 + PP(1,*)^2 + PP(2,*)^2)

ind=where(r lt rad)
r=r(ind) * Time
r=r(sort(r))

m= PartMass*(lindgen(n_Elements(r))+1)

delta= m/(4*!PI/3.0 * r^3) 


ind=where(delta ge 200*rhocrit)
r200 = max(r(ind))
m200 = 200* 4*!PI/3 * r200^3 * rhocrit


print, "R200=", R200, "  M200=", M200, "  V200=", sqrt(43.0071*m200/r200)




mi= 0.001
ma= 5.0

BINS = 25

countarr = lindgen(BINS)

for i=0L, n_elements(r)-1 do begin

  b = (alog(r(i))-alog(mi))/(alog(ma)-alog(mi)) * BINS
  if (b ge 0) and (b lt BINS) then begin
     countarr(b) = countarr(b)+1
  endif

endfor

rbin = exp((indgen(BINS)+0.5)* (alog(ma)-alog(mi))/BINS + alog(mi))
r1 = exp((indgen(BINS)+0.0)* (alog(ma)-alog(mi))/BINS + alog(mi))
r2 = exp((indgen(BINS)+1.0)* (alog(ma)-alog(mi))/BINS + alog(mi))

rho = countarr*PartMass/(4*!PI/3.0 * (r2^3-r1^3))


plot, rbin, rho/rhocrit, /xlog,/ylog, yrange=[1,1.0e6], psym=5


oplot, 2*time*[0.005,0.005],[0.0001, 1.0e9], linestyle=1


dd = rho/rhocrit


Rs=0.01

ind=where((dd gt 20.0) and (Rbin gt 0.5*0.005*time))

R= Rbin(ind)
Rho= rho(ind)

Bins = n_elements(ind)

for rep=0,200 do begin

    logRho0 = total(alog(R/Rs)+2*alog(1+R/Rs)+alog(Rho))/Bins

    repeat begin


        S=total( ( logRho0 - alog(R/Rs)- 2*alog(1+R/Rs)- alog(Rho))^2)

        dS= 2*total( ( logRho0 -alog(R/rs)-2*alog(1+R/Rs)-alog(Rho))*(1/Rs + 2*R/Rs^2/(1+R/Rs)))

        ddS= 2*total( ( logRho0 -alog(R/rs)-2*alog(1+R/Rs)-alog(Rho))* $
                      (-1/Rs^2 - 2*R/(RS*(Rs+R))^2 *(2*Rs+R)) $
                      + (1/Rs + 2*R/Rs^2/(1+R/Rs))^2 )

        DRs= - DS/ddS

        Rs=Rs + DRs

;        print,S, rs, drs
    endrep until abs(dRs/Rs) le 1.0e-8

endfor


Rfit=[0.0001, R, 10.0]

fit=exp(logRho0)/ ( Rfit/Rs * (1+Rfit/Rs)^2)

oplot, Rfit, Fit/rhocrit


print, "CC=" , R200/rs, "  rs=", rs

endfor


end

