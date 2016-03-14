
;;; we will do a band averaging of the finely binned raw powerspectrum
;;; points, subject to two conditions:
;;; We want enough modes per bin in order to reduce the variance in a bin,
;;; and simultaneously, for large k, we don't want the bins become too narrow.
;;;
;;; The first condition is set by "MinModeCount",
;;; the second by "TargetBinNummer", which is used to compute a minimum 
;;; logarithmic bin-size.

MinModeCount =    140
TargetBinNummer = 180


;;; Define minimum and maximum for the axes-ranges that are plotted

ymin = -4.0     
ymax =  4.0

kmin = 0.005
kmax = 400.0


;;; Alternative for full range, for example: 

;; MinModeCount = 3
;; TargetBinNummer = 100
;; kmin = 0.005   
;; kmax = 400.0
;; ymin = -3.0     
;; ymax = 2.2


;;; Set the number of the output, and the pathname to the
;;; powerspectrum measurements

Num = 63
BaseDir =   "/afs/mpa/sim/Millennium/powerspec/"    



;;;;  Read the powerspectrum measurement

exts='000'
exts=exts+strcompress(string(num),/remove_all)
exts=strmid(exts,strlen(exts)-3,3)
fname = Basedir + "powerspec_"+exts+".txt"

openr, 1, fname
Time = 0.0 
Bins = 0L
readf, 1, Time
readf, 1, Bins
da1= fltarr(10, bins)
readf, 1, da1
readf, 1, Time
readf, 1, Bins
da2= fltarr(10, bins)
readf, 1, da2
close,1


;;;;  Extract the two measured pieces from the table read in

K_A = da1(0,*)
Delta2_A = da1(1,*)
Shot_A = da1(2,*)
ModePow_A = da1(3,*)
ModeCount_A = da1(4,*)
Delta2Uncorrected_A = da1(5,*)
ModePowUncorrected_A = da1(6,*)
Specshape_A =  da1(7,*)
SumPower_A =  da1(8,*)
ConvFac_A =  da1(9,*)

K_B = da2(0,*)
Delta2_B = da2(1,*)
Shot_B = da2(2,*)
ModePow_B = da2(3,*)
ModeCount_B = da2(4,*)
Delta2Uncorrected_B = da2(5,*)
ModePowUncorrected_B = da2(6,*)
Specshape_B =  da2(7,*)
SumPower_B =  da2(8,*)
ConvFac_B =  da2(9,*)



;;; Do the (re)binning of the spectrum

MinDlogK = (alog10(max(K_A)) - alog10(min(K_A)))/TargetbinNummer

istart=0
ind=[istart]
k_list_A = [0]
Delta2_list_A = [0]
count_list_A = [0]
repeat begin
    count = total(modecount_a(ind))
    deltak =  (alog10(max(K_A(ind))) - alog10(min(K_A(ind))))
    if (deltak ge mindlogk) and (count ge MinModeCount) then begin
        d2 = total(SumPower_A(ind))/total(ModeCount_A(ind)) 
        b = fix(total(double(ind)*ModeCount_A(ind))/total(ModeCount_A(ind)))
        kk = K_A(b)
        d2 = ConvFac_A(b)*d2*Specshape_A(b)
        k_list_A = [k_list_A, kk]
        Delta2_list_A = [Delta2_list_A, d2]
        count_list_A = [count_list_A, total(ModeCount_A(ind))]
        istart = istart + 1
        ind = [istart]
    endif else begin
        istart = istart + 1
        ind = [ind, istart]
    endelse
endrep until istart ge Bins
K_list_A = k_list_A(1:*)
Delta2_list_A = delta2_list_A(1:*)
Count_list_A = count_list_A(1:*)



istart=0
ind=[istart]
k_list_B = [0]
Delta2_list_B = [0]
count_list_B = [0]
repeat begin
    count = total(modecount_B(ind))
    deltak =  (alog10(max(K_B(ind))) - alog10(min(K_B(ind))))
    if (deltak ge mindlogk) and (count ge MinModeCount) then begin
        d2 = total(SumPower_B(ind))/total(ModeCount_B(ind)) 
        b = fix(total(double(ind)*ModeCount_B(ind))/total(ModeCount_B(ind)))
        kk = K_B(b)
        d2 = ConvFac_B(b) * d2 * Specshape_B(b)
        k_list_B = [k_list_B, kk]
        Delta2_list_B = [Delta2_list_B, d2]
        count_list_B = [count_list_B, total(ModeCount_B(ind))]
        istart = istart + 1
        ind = [istart]
    endif else begin
        istart = istart + 1
        ind = [ind, istart]
    endelse
endrep until istart ge Bins
K_list_B = k_list_B(1:*)
Delta2_list_B = delta2_list_B(1:*)
Count_list_B = count_list_B(1:*)



;;; Get the growth-factor corresponding for the current time

A= Time
Redshift = 1/A - 1
Lines=110
openr,1,"growthfac.txt"
da=dblarr(2,Lines)
readf,1,da
close,1

time  =   da(0,*)
dplus  =  da(1,*)

;;; Interpolate the from the growthfactor table
;;; (Note that this does not work in IDL 5.3 for some reason!)

growthfac = exp( interpol( alog(dplus), alog(time), alog(a)))
growthfac = growthfac(0)


;;; Get the linear input spectrum

fname = "inputspec_ics_millennium.txt"
spawn,"wc "+fname,result
result=strtrim(result, 1)
lines=long(result)
lines=lines(0)
en=fltarr(4, lines-1)
openr,1,fname
zstart=0.0
growthfactor= 0.0
readf,1,zstart, growthfactor
readf,1,en
close,1
k_linear =  en(0,*)
d2_linear = en(1,*)
ind=where (d2_linear gt 0)
k_linear = k_linear(ind)
d2_linear = d2_linear(ind)


;;; Add an extrapolation to the linear input spectrum into unsampled
;;; small scales, for plotting purposes

k1= k_linear(n_elements(k_linear)-2)
k2= k_linear(n_elements(k_linear)-1) 
d1= d2_linear(n_elements(d2_linear)-2) 
d2= d2_linear(n_elements(d2_linear)-1) 
k_ext= 400.0
d2_ext = exp(alog(d2/d1)/alog(k2/k1) * alog(k_ext/k2) + alog(d2))
k_linear = [k_linear, k_ext]
d2_linear = [d2_linear, d2_ext]


;;;; Scale the linear spectrum to the present time

d2_linear = d2_linear / growthfac^2





;;;; Make a postscript plot

fout=  "ps_"+exts+".eps"

mydevice=!d.name
set_plot,'PS'

!p.font=0
device,/times,/italic,font_index=20
device,xsize=16.0,ysize=14.0
!x.margin=[9,3]
!p.thick=2.5
!p.ticklen=0.03

device,filename= fout, /encapsulated, /color

v1=[255,150,  0,0,0 ,  255, 100,240, 230]
v2=[  0, 0,  0,150,255, 255, 200,240,230]
v3=[  0,  0,255,0,255, 0,   50,240,230]
tvlct,v1,v2,v3,1


;;; Create the axes and labels

plot, [100], [100], psym=3, /xlog,  $
  xtitle = "!20k!7 [ !20h!7 / Mpc ]!3", ytitle = "!7log !9D!7!U2!N (!20k!7)!3 ",  $
  xrange=[kmin, kmax], yrange=[ymin, ymax], $
  charsize=1.1, xthick=3.0, ythick=3.0, xstyle=1,ystyle=1


;;; Plot the linear power spectrum 

oplot, k_linear, alog10(D2_linear), thick=3.0


;;; Evalute the shot-noise at the positions of the bins

shot_list_A = Shot_a(0) * (k_list_a/k_a(0))^3
shot_list_B = Shot_b(0) * (k_list_b/k_b(0))^3


;;; do (or not do) the shot noise substraction

;Delta2_list_B = Delta2_list_B  - shot_list_b
;Delta2_list_A = Delta2_list_A  - shot_list_a


;;; plot the shot-noise limit

oplot, k_b, alog10(shot_b), linestyle=1


;;; Get the number of independent modes in the first part of the
;;; spectrum (note: Each mode was counted twice by the measurement code)

count = count_list_b/2.0

for i = 0, n_elements(count)-1 do begin
    min = 1.0 - 1.0/sqrt(count(i))
    max = 1.0 + 1.0/sqrt(count(i))
    if count(i) lt 1.0e4 then begin
        errplot, [K_list_B(i)] , alog10([Delta2_list_B(i)] * min ) , alog10([Delta2_list_B(i)] * max ) 
    endif
endfor

;;; As an alternative to the above quick & dirty error bars, the 
;;; following will map out the distribution of mode amplitudes
;;; and draw the corresponding asymmetric error bars for the expected
;;; Rayleigh diistribution

;for i = 0, 6 do begin
   
;    c= count(i)

;    NN = 10000    
;    x=0
;    for rep = 1,c do begin
;        r=randomu(seed, NN)
;        rr = -alog(r)
;        x = x + rr
;    endfor
;    x = x/c
;    x = x(sort(x))
    
;    min = x(0.16*NN)
;    max = x(0.84*NN)
    
;    errplot, [K_list_B(i)] , alog10([Delta2_list_B(i)] * min /K_list_B(i)) , alog10([Delta2_list_B(i)] * max / K_list_B(i))  
;endfor







;;; define in which k-ranges each of the two power spectrum
;;; measurements should be used

indB=where((K_list_B lt 5) and (k_list_B ge 0.01))
indA=where((K_list_A ge 5) and (k_list_A le 200))


;;; make a list with the points we want to plot

kk = [K_list_B(indB), K_list_A(indA)]
pp = [Delta2_list_B(indb),Delta2_list_A(indA)]

shot_pp = Shot_b(0) * (kk/k_b(0))^3

;;; select the region where we not significantly below the shot noise

ind= where(pp gt 0.5*shot_pp)

;;; define a circle as 

x=cos(2*!PI*indgen(16)/15.0)  * 0.5
y=sin(2*!PI*indgen(16)/15.0)  * 0.5
usersym, x,y ,/fill  ;   circle


;;; Plot the power spectrum measurement with some dots. 
;;; Note: We plot D^2(k) / k here to compress the y-range a bit

;oplot, kk(ind), alog10(pp(ind)), color=1, psym=8, symsize=1.1
oplot, K_list_B(indB), alog10(Delta2_list_B(indb)), color=1, psym=8, symsize=1.1
oplot, K_list_A(indA), alog10(Delta2_list_A(inda)), color=3, psym=8, symsize=1.1



xyouts, 0.20,0.85,/normal, "!7z = "+string(format='(F6.2)',Redshift)+"!3",  charsize=1.2, color=1

device,/close
set_plot,"X"


end
