
fname = "/ptmp/vrs/Millennium/correl_041.txt"

openr, 1, fname
Time = 0.0 & readf, 1, time
Bins = 0L & readf, 1, bins
da= fltarr(4, bins)
readf, 1, da
close, 1


R = da(0,*)
Xi = da(1,*)
Pairs = da(2,*)
Shells = da(3,*)

plot, R, Xi, /xlog,/ylog, yrange=[1.0e-2,4000]

end
