
BaseDir =   "/gmpa/mpa/vrs/test216f/"    ; The output-directory of the simulation
SnapBase=   "snapshot"                   ; The basename of the snapshot files

Num = 1                          ; The number of the snapshot we look at


;;; The object-file of the compile C-library for accessing the group catalogue

ObjectFile = "/gmpa/mpa/vrs/test216f/L-Gadget2/IdlGroupLib/idlgrouplib.so"

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

window, xsize=1000, ysize=1000
!p.multi=[0,2,2]

for rep=0,3 do begin


  N= 2+rep  ; group number

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


   Plot, Pos(0,*), Pos(1,*), psym=3

   print, "GroupNr=", N, " length=", Len, "   center of mass=", sx, sy, sz

endfor

end
