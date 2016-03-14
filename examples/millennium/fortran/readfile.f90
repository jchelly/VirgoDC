MODULE readfilemod
!
! Routines to read L-Gadget2 particle data files in Fortran
!
! Which file to read is specified by setting -
!
! CHARACTER(LEN=*) :: basedir - the directory containing the postproc
!                                and snapdir directories
!
! If basedir is set to 'MILLENNIUM', it is assumed that the files are to be
! read from the Durham millennium data store and appropriate file paths
! are generated automatically.
!
! INTEGER :: isnap             - the snapshot number (ie which output time)
! INTEGER :: ifile             - which file within the snapshot
!
! Subgroup files are read from the basedir/postproc_???/ directory and
! friends of friends files are read from basedir/snapdir_???/
!
! All array parameters should be declared with the POINTER attribute by
! the calling program and should not have been allocated or made to point
! at anything before these routines are called (this is checked).
!
!
!
!
! Data type corresponding to gadget file header 
  TYPE headertype
     INTEGER*4, DIMENSION(6) :: npart
     REAL*8, DIMENSION(6) :: mass
     REAL*8 :: time
     REAL*8 :: redshift
     INTEGER*4 :: flag_sfr
     INTEGER*4 :: flag_feedback
     INTEGER*4, DIMENSION(6) :: nparttotal
     INTEGER*4 :: flag_cooling
     INTEGER*4 :: numfiles
     REAL*8 :: boxsize
     REAL*8 :: omega0
     REAL*8 :: omegalambda
     REAL*8 :: hubbleparam
     INTEGER*4 :: flag_stellarage
     INTEGER*4 :: flag_metals
     INTEGER*4 :: hashtabsize
     CHARACTER, DIMENSION(84) :: unused
  END TYPE headertype

CONTAINS

! ---------------------------------------------------------------------------

  SUBROUTINE readheader(basedir,basename,isnap,ifile,header)
!
! Read and return the gadget file header for the specified file
!
    USE filepathmod
    IMPLICIT NONE
! Input parameters
    CHARACTER(LEN=*), INTENT(IN) :: basedir,basename
    INTEGER, INTENT(IN) :: isnap, ifile
! Header to return
    TYPE (headertype) :: header
! Internal variables
    CHARACTER(LEN=256) :: filename
    INTEGER :: lnblnk

    IF(basedir.EQ.'MILLENNIUM')THEN
       filename=filepath(0,isnap,ifile)
    ELSE
       IF(ifile.GT.99)THEN
          WRITE(filename,'(a,"/snapdir_",i3.3,"/",a,"_",i3.3,".",i3.3)') &
               basedir(1:lnblnk(basedir)),isnap,basename(1:lnblnk(basename)), &
               isnap,ifile
       ELSE IF(ifile.GT.9)THEN
          WRITE(filename,'(a,"/snapdir_",i3.3,"/",a,"_",i3.3,".",i2.2)') &
               basedir(1:lnblnk(basedir)),isnap,basename(1:lnblnk(basename)), &
               isnap,ifile
       ELSE
          WRITE(filename,'(a,"/snapdir_",i3.3,"/",a,"_",i3.3,".",i1.1)') &
               basedir(1:lnblnk(basedir)),isnap,basename(1:lnblnk(basename)), &
               isnap,ifile
       END IF
    END IF

    OPEN(unit=1,file=filename,status='old',action='read',form='unformatted')
    ! Byte swapping doesn't work if you just do READ(1)header
    READ(1)header%npart,header%mass,header%time,header%redshift, &
         header%flag_sfr,header%flag_feedback,header%nparttotal, &
         header%flag_cooling,header%numfiles,header%boxsize, &
         header%omega0,header%omegalambda,header%hubbleparam, &
         header%flag_stellarage,header%flag_metals,header%hashtabsize
    CLOSE(1)

  END SUBROUTINE readheader

! ---------------------------------------------------------------------------

  SUBROUTINE readfile(basedir,basename,isnap,ifile,header,firstcell, &
       lastcell,hashtable,pos,vel,id)
!
! Read and return all data from the specified file. Output arrays must
! already be allocated. Use readheader to get particle numbers to do this.
!
    USE filepathmod
    IMPLICIT NONE
! Input parameters
    CHARACTER(LEN=*), INTENT(IN) :: basedir,basename
    INTEGER, INTENT(IN) :: isnap, ifile
! Header and hash table to return
    TYPE (headertype) :: header
    INTEGER :: firstcell, lastcell
    INTEGER, DIMENSION(*) :: hashtable
! Particle data
    REAL, DIMENSION(3,*) :: pos,vel
    INTEGER*8, DIMENSION(*) :: id
! Internal variables
    CHARACTER(LEN=256) :: filename
    INTEGER :: lnblnk
    INTEGER :: np, nhash
    INTEGER, DIMENSION(2) :: h

    IF(basedir.EQ.'MILLENNIUM')THEN
       filename=filepath(0,isnap,ifile)
    ELSE
       IF(ifile.GT.99)THEN
          WRITE(filename,'(a,"/snapdir_",i3.3,"/",a,"_",i3.3,".",i3.3)') &
               basedir(1:lnblnk(basedir)),isnap,basename(1:lnblnk(basename)), &
               isnap,ifile
       ELSE IF(ifile.GT.9)THEN
          WRITE(filename,'(a,"/snapdir_",i3.3,"/",a,"_",i3.3,".",i2.2)') &
               basedir(1:lnblnk(basedir)),isnap,basename(1:lnblnk(basename)), &
               isnap,ifile
       ELSE
          WRITE(filename,'(a,"/snapdir_",i3.3,"/",a,"_",i3.3,".",i1.1)') &
               basedir(1:lnblnk(basedir)),isnap,basename(1:lnblnk(basename)), &
               isnap,ifile
       END IF
    END IF

    OPEN(unit=1,file=filename,status='old',action='read',form='unformatted')
    ! Byte swapping doesn't appear to work if you just do READ(1)header
    READ(1)header%npart,header%mass,header%time,header%redshift, &
         header%flag_sfr,header%flag_feedback,header%nparttotal, &
         header%flag_cooling,header%numfiles,header%boxsize, &
         header%omega0,header%omegalambda,header%hubbleparam, &
         header%flag_stellarage,header%flag_metals,header%hashtabsize 
    np=header%npart(2)
    READ(1)pos(1:3,1:np)
    READ(1)vel(1:3,1:np)
    READ(1)id(1:np)
    READ(1)h(1:2)
    firstcell=h(1)
    lastcell=h(2)
    nhash=lastcell-firstcell+1
    READ(1)hashtable(1:nhash)
    CLOSE(1)

  END SUBROUTINE readfile

END MODULE readfilemod

