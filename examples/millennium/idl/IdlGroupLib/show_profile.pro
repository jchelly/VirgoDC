
BaseDir =   "/virgo/data/Millennium/"    ; The output-directory of the simulation
SnapBase=   "snap_millennium"                   ; The basename of the snapshot files

Num = 30                          ; The number of the snapshot we look at

Nr = 0                      ; The group number


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


;;;;;; Now we are all set to read out individual groups (repeatedly if desired)

;;; In this example, we show the 3rd, 4th, 5th and 6th most massive groups.



  N= Nr  ; group number

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



  x= Pos(0,*)
  y= Pos(1,*)
  z= Pos(2,*)

  r=sqrt(x^2 + y^2 + z^2)
  rmax = max(r)

  repeat begin
      r=sqrt(x^2 + y^2 + z^2)
      ind = where(r lt rmax)

      if n_elements(ind) ge 15 then begin
          xc= total(x(ind), /double)/n_elements(ind)
          yc= total(y(ind), /double)/n_elements(ind)
          zc= total(z(ind), /double)/n_elements(ind)

          x= x -xc
          y= y -yc
          z= z -zc

          print, n_elements(ind), xc, yc, zc
      endif

      rmax = 0.9 * rmax
  endrep until n_elements(ind) lt 15

  r=sqrt(x^2 + y^2 + z^2)

  m= lindgen(n_Elements(r))+1

  r=r(sort(r))

  v= sqrt(m/r)

  plot, r, v, psym=3, /xlog, /ylog


end
