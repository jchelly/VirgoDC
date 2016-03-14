
BaseDir =   "/ptmp/vrs/Millennium/"    
;BaseDir =   "/r/v/vrs/Millennium/"    
SnapBase=   "snap_millennium"                   ; The basename of the snapshot files

Num = 37                     ; The number of the snapshot we look at

;;; The object-file of the compile C-library for accessing the group catalogue

ObjectFile = "/u/vrs/Millennium/L-Gadget2/IdlGroupLib/idlgrouplib2.so"


Num=long(Num)
exts='000'
exts=exts+strcompress(string(num),/remove_all)
exts=strmid(exts,strlen(exts)-3,3)
Outputdir  = Basedir + "/snapdir_"+exts+"/"



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


  TabLen = lonarr(512)

  result = CALL_EXTERNAL(ObjectFile, $
                       'get_particles_in_cubes', /UL_VALUE, $
                        OutputDir, $
                        Num, $
                        SnapBase, $
                        HashTable, $
                        FileTable, $
                        HashTabSize, $
                        LastHashCell, $
			NInFiles, $
                        NFiles, $ 
                        TabLen)


plot, tablen

end
