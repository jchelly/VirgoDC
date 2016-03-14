;goto,jump

BaseDir =   "/ptmp/vrs/Millennium/"    
;BaseDir =   "/r/v/vrs/Millennium/"    
SnapBase=   "snap_millennium"                   ; The basename of the snapshot files



Num = 39                     ; The number of the snapshot we look at

;;; The object-file of the compile C-library for accessing the group catalogue

ObjectFile = "/u/vrs/Millennium/L-Gadget2/IdlGroupLib/idlgrouplib2.so"


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






window, xsize=1100, ysize=1100

!P.multi=[0,3,3]


for GGN=0L, 8 do begin

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

jump:

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


          print, n_elements(ind), xxc, yyc, zzc
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


;ind=where(abs(PP(2,*)) lt 500.0)

;plot, PP(0,ind), PP(1,ind), psym=3, xrange=[-1,1]/2.0, yrange=[-1,1]/2.0

;ind=where(abs(Pos(2,*)-zzc) lt 500.0)

;oplot, Pos(0,ind)-xxc, Pos(1,ind)-yyc, psym=3, color=255



r=sqrt(PP(0,*)^2 + PP(1,*)^2 + PP(2,*)^2)

ind=where(r lt rad)
r=r(ind)

m= lindgen(n_Elements(r))+1

r=r(sort(r))

v= sqrt(m/r)

plot, r, v, psym=3, /xlog, /ylog

endfor


end
