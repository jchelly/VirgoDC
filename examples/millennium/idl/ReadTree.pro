
Base = "/ptmp/vrs/Millennium/"


HaloStruct = {$        
               Descendant           : 0L, $
               FirstProgenitor      : 0L, $
               NextProgenitor       : 0L, $
               FirstHaloInFOFgroup  : 0L, $
               NextHaloInFOFgroup   : 0L, $
               Len                  : 0L, $
               M_Mean200            : 0.0, $
               M_Crit200            : 0.0, $ 
               M_TopHat             : 0.0, $
               Pos                  : fltarr(3), $
               Vel                  : fltarr(3), $
               VelDisp              : 0.0, $
               Vmax                 : 0.0, $
               Spin                 : fltarr(3), $
               MostBoundID          : lon64arr(1), $
               SnapNum              : 0L, $ 
               FileNr               : 0L, $
               SubhaloIndex         : 0L, $
               SubhalfMass          : 0.0 $
             }

Num = 63

for FileNr = 0, 511 do begin

    print, "Doing FileNr= ", FileNr

    exts='000'
    exts=exts+strcompress(string(num),/remove_all)
    exts=strmid(exts,strlen(exts)-3,3)


    fname= base+"/treedata/trees_"+ exts + ". " + string(FileNr)
    fname=strcompress(fname,/remove_all)

    openr,1, fname
    Ntrees = 0L
    TotNHalos = 0L
    readu,1,Ntrees,TotNhalos
    print,"Ntrees= ", Ntrees
    print,"TotNhalos= ", TotNhalos
    TreeNHalos = lonarr(Ntrees)
    readu,1,TreeNhalos

    for tr=0L, Ntrees-1 do begin

        Tree = replicate(HaloStruct, TreeNhalos(tr))
        readu,1, Tree

    endfor  
    close,1

endfor

end
