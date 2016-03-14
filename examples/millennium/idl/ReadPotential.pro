
Base= "/ptmp/vrs/milliMillennium//"

Num = 42  ; number of snapshot files


exts='000'
exts=exts+strcompress(string(Num),/remove_all)
exts=strmid(exts,strlen(exts)-3,3)

subfile = 0 
slabscount = 0L

repeat begin

    f= Base + "/snapdir_"+exts+"/potential_"+exts +"."+string(subfile)
    f= Strcompress(f, /remove_all)

    grid= 0L
    sizeof= 0L
    slabspertask = 0L
    firstslab = 0L
    boxsize= 0.0D
    asmth = 0.0D

    openr,1,f
    readu,1, grid, sizeof, slabspertask, firstslab, boxsize, asmth
    pot = fltarr(grid, grid, slabspertask)
    readu,1,pot
    close,1

    if subfile eq 0 then begin
        potential = fltarr(GRID, GRID, GRID)
    endif

    potential(0:Grid-1, 0:Grid-1, firstslab:firstslab+slabspertask-1) = pot(*,*,*)
    slabscount = slabscount + slabspertask
    subfile = subfile + 1

endrep until slabscount ge GRID

potential = transpose(potential)

tvscl, potential(*,*,0)


end

















 
