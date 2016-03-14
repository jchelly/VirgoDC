PROGRAM readsubhalosexample
!
! Read the subhalos from one file and output half mass radius and x,y,z
! coordinates 
!
! Also outputs m200,r200,x,y,z for each FoF halo to another file - note that
! FoF groups with no subgroups are not included as subfind doesn't record
! positions for these groups.
!
! This can be adapted to output any of the other SubFind group properties
! just by editing the WRITE statements near the end of the file. See
! readgroups.f90 for the definition of the halo/subhalo data types.
!
  USE readgroupsmod
  IMPLICIT NONE
! Parameters
  CHARACTER(LEN=500) :: basedir, basename, outfile1,outfile2
  INTEGER :: isnap,nfile,ifile
! Groups
  INTEGER, DIMENSION(:), POINTER :: nsphalo,firstsub,sublen,suboffset,subpar
  TYPE (subhalotype), DIMENSION(:), POINTER :: subhalo
  TYPE (halotype), DIMENSION(:), POINTER :: halo
  INTEGER :: nfoffile,nsubfile,nfoftot,nidfile
! Loop index
  INTEGER :: isub, ifof

! Get parameters
  WRITE(*,*)'Enter number of file to read'
  READ(*,*)ifile
  WRITE(*,*)'Enter name of directory with snapdir and postproc subdirectories'
  READ(*,'(a)')basedir
  WRITE(*,*)'Enter snapshot basename'
  READ(*,'(a)')basename
  WRITE(*,*)'Enter number of snapshot to read'
  READ(*,*)isnap
  WRITE(*,*)'Enter name of output file for halos'
  READ(*,'(a)')outfile1
  WRITE(*,*)'Enter name of output file for subhalos'
  READ(*,'(a)')outfile2
  
! Read in the groups from file ifile
  WRITE(*,*)'Reading groups'
  CALL readsubhalos(basedir,isnap,ifile,nfoffile,nsubfile, &
       nidfile,nfoftot,nfile,halo,subhalo)

! Output coordinates, radius and mass of each subgroup
! (radius and mass defined such that density = 200*universal mean density)
  OPEN(unit=1,file=outfile2,status='unknown',form='formatted')
  WRITE(1,'(5a16)') &
       'No.of particles','Rhalf (Mpc/h)','x (Mpc/h)','y (Mpc/h)','z (Mpc/h)'
  DO isub=1,nsubfile,1
     WRITE(1,'(i16,5e16.8)') &
          subhalo(isub)%len, &
          subhalo(isub)%halfmass,subhalo(isub)%pos(1), &
          subhalo(isub)%pos(2),subhalo(isub)%pos(3)
  END DO
  CLOSE(1)

! Output coordinates and r200 for each halo and mass within r200
! (radius in which density is 200 times universal mean density)
! Position of halo is taken to be position of most bound particle of most
! massive subgroup
  OPEN(unit=1,file=outfile1,status='unknown',form='formatted')
  WRITE(1,'(5a16)')'m200 (Msolar/h)','r200 (Mpc/h)','x (Mpc/h)', &
       'y (Mpc/h)','z (Mpc/h)'
  DO ifof=1,nfoffile
     IF(halo(ifof)%nsubgroups.GT.0)THEN
        WRITE(1,'(5e16.8)') &
             halo(ifof)%mmean200*1.0e10,halo(ifof)%rmean200, &
             subhalo(halo(ifof)%firstsubgroup+1)%pos(1), &
             subhalo(halo(ifof)%firstsubgroup+1)%pos(2), &
             subhalo(halo(ifof)%firstsubgroup+1)%pos(3)
     END IF
  END DO
  CLOSE(1)

END PROGRAM readsubhalosexample

