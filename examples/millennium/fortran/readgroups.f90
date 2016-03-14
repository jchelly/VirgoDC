MODULE readgroupsmod
!
! This assumes that unformatted record lengths are specified in bytes,
! so on duomo this file needs to be compiled with the '-assume byterecl' flag.
! The Linux (pgf90) and Sun compilers seem to do this by default.
!
! Routines to read the millennium group files using Fortran direct access
!
!  - readgrouptab
!  - readgroupids
!  - readsubtab
!  - readsubids
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
! Note that groupids must be declared as INTEGER*8 because the IDs need
! to be stored as long integers.
!
!
! Types to describe halos and subhalos - this is to avoid having a subroutine
! which returns a huge number of parameters if you want to read all of the
! (sub)halo properties. Only used by readsubhalos().
!
! Type to describe a single subhalo
TYPE subhalotype
    REAL, DIMENSION(3) :: pos, vel, spin
    REAL :: veldisp
    REAL :: vmax
    INTEGER*8 :: mostboundid
    REAL :: halfmass
    INTEGER :: len
    INTEGER :: offset
    INTEGER :: parent
END TYPE subhalotype
! Type to describe a single halo
TYPE halotype
    REAL :: mmean200
    REAL :: rmean200
    REAL :: mcrit200
    REAL :: rcrit200
    REAL :: mtophat200
    REAL :: rtophat200
    INTEGER :: nsubgroups
    INTEGER :: firstsubgroup
END TYPE halotype

CONTAINS
!
! -----------------------------------------------------------------------
!
  SUBROUTINE readgrouptab(basedir,isnap,ifile,nfoffile,nidfile,nfoftot, &
       nfiles,foflen,foffset)
!
! Read a single group_tab file
!
! Parameters
    USE filepathmod
    IMPLICIT NONE
    CHARACTER(LEN=*), INTENT(IN) :: basedir
    INTEGER, INTENT(IN) :: ifile, isnap
    INTEGER, INTENT(OUT) :: nfoffile,nfoftot,nfiles,nidfile
    INTEGER, DIMENSION(:), POINTER :: foflen, foffset
! Internal variables
    CHARACTER(LEN=256) :: filename
    INTEGER, DIMENSION(4) :: header
    INTEGER :: nint, nbytes
    INTEGER :: lnblnk
!
! Check fofflen/foffset pointers not pointing to anything
!
    IF(ASSOCIATED(foflen).OR.ASSOCIATED(foffset)) &
         STOP 'Input pointers already allocated!'
!
! Generate file name
!
    IF(basedir.EQ.'MILLENNIUM')THEN
       filename=filepath(1,isnap,ifile)
    ELSE
       IF(ifile.GT.99)THEN
          WRITE(filename,'(a,"/snapdir_",i3.3,"/group_tab_",i3.3,".",i3.3)') &
               basedir(1:lnblnk(basedir)),isnap,isnap,ifile
       ELSE IF (ifile.GT.9)THEN
          WRITE(filename,'(a,"/snapdir_",i3.3,"/group_tab_",i3.3,".",i2.2)') &
               basedir(1:lnblnk(basedir)),isnap,isnap,ifile
       ELSE
          WRITE(filename,'(a,"/snapdir_",i3.3,"/group_tab_",i3.3,".",i1.1)') &
               basedir(1:lnblnk(basedir)),isnap,isnap,ifile
       END IF
    END IF
!
! Read the number of groups
!
    OPEN(UNIT=1,FILE=filename,STATUS='old',ACCESS='direct',RECL=20, &
         ACTION='read')
    READ(UNIT=1,REC=1)header
    CLOSE(1)
    nfoffile=header(1)
    nidfile=header(2)
    nfoftot=header(3)
    nfiles=header(4)
!
! Allocate storage then read the file
!
    nint=4+(nfoffile*2)
    nbytes=nint*4
    ALLOCATE(foflen(nfoffile),foffset(nfoffile))
    OPEN(UNIT=1,FILE=filename,STATUS='old',ACCESS='direct',RECL=nbytes, &
         ACTION='read')
    READ(UNIT=1,REC=1)header,foflen,foffset
    CLOSE(1)
!
! Finished!
!
    RETURN
  END SUBROUTINE readgrouptab
!
! -----------------------------------------------------------------------
!
  SUBROUTINE readgroupids(basedir,isnap,ifile,nfoffile,nidfile,nfoftot, &
       nfiles,groupids)
!
! Read a single group_tab file
!
! Parameters
    USE filepathmod
    IMPLICIT NONE
    CHARACTER(LEN=*), INTENT(IN) :: basedir
    INTEGER, INTENT(IN) :: ifile, isnap
    INTEGER, INTENT(OUT) :: nfoffile,nfoftot,nfiles,nidfile
    INTEGER*8, DIMENSION(:), POINTER :: groupids
! Internal variables
    CHARACTER(LEN=256) :: filename
    INTEGER, DIMENSION(4) :: header
    INTEGER :: nint8, nbytes
    INTEGER :: lnblnk
!
! Check groupids pointer not pointing to anything
!
    IF(ASSOCIATED(groupids)) &
         STOP 'Input pointers already allocated!'
!
! Generate file name
!
    IF(basedir.EQ.'MILLENNIUM')THEN
       filename=filepath(4,isnap,ifile)
    ELSE
       IF(ifile.GT.99)THEN
          WRITE(filename,'(a,"/snapdir_",i3.3,"/group_ids_",i3.3,".",i3.3)') &
               basedir(1:lnblnk(basedir)),isnap,isnap,ifile
       ELSE IF (ifile.GT.9)THEN
          WRITE(filename,'(a,"/snapdir_",i3.3,"/group_ids_",i3.3,".",i2.2)') &
               basedir(1:lnblnk(basedir)),isnap,isnap,ifile
       ELSE
          WRITE(filename,'(a,"/snapdir_",i3.3,"/group_ids_",i3.3,".",i1.1)') &
               basedir(1:lnblnk(basedir)),isnap,isnap,ifile
       END IF
    END IF
!
! Read the number of groups
!
    OPEN(UNIT=1,FILE=filename,STATUS='old',ACCESS='direct',RECL=16, &
         ACTION='read')
    READ(UNIT=1,REC=1)header
    CLOSE(1)
    nfoffile=header(1)
    nidfile=header(2)
    nfoftot=header(3)
    nfiles=header(4)
!
! Allocate storage then read the file
!
    nint8=2+nidfile
    nbytes=nint8*8
    ALLOCATE(groupids(nidfile))
    OPEN(UNIT=1,FILE=filename,STATUS='old',ACCESS='direct',RECL=nbytes, &
         ACTION='read')
    READ(UNIT=1,REC=1)header,groupids
    CLOSE(1)
!
! Finished!
!
    RETURN
  END SUBROUTINE readgroupids

!
! -----------------------------------------------------------------------
!
  SUBROUTINE readsubtab(basedir,isnap,ifile,nfoffile,nsubfile, &
       nidfile,nfoftot,nfiles,nsphalo,firstsub,sublen,suboffset,subpar)
!
! Read a single sub_tab file
!
! Parameters
    USE filepathmod
    IMPLICIT NONE
    CHARACTER(LEN=*), INTENT(IN) :: basedir
    INTEGER, INTENT(IN) :: ifile, isnap
    INTEGER, INTENT(OUT) :: nfoffile,nfoftot,nfiles,nidfile,nsubfile
    INTEGER, DIMENSION(:), POINTER :: sublen,suboffset,subpar,nsphalo,firstsub
! Internal variables
    CHARACTER(LEN=256) :: filename
    INTEGER, DIMENSION(5) :: header
    INTEGER :: nint, nbytes
    INTEGER :: lnblnk
    INTEGER :: offset
!
! Check sublen/suboffset pointers not pointing to anything
!
    IF(ASSOCIATED(sublen).OR.ASSOCIATED(suboffset).OR.ASSOCIATED(subpar) &
         .OR.ASSOCIATED(nsphalo).OR.ASSOCIATED(firstsub)) &
         STOP 'Input pointers already allocated!'
!
! Generate file name
!
    IF(basedir.EQ.'MILLENNIUM')THEN
       filename=filepath(2,isnap,ifile)
    ELSE
       IF(ifile.GT.99)THEN
          WRITE(filename,'(a,"/postproc_",i3.3,"/sub_tab_",i3.3,".",i3.3)') &
               basedir(1:lnblnk(basedir)),isnap,isnap,ifile
       ELSE IF (ifile.GT.9)THEN
          WRITE(filename,'(a,"/postproc_",i3.3,"/sub_tab_",i3.3,".",i2.2)') &
               basedir(1:lnblnk(basedir)),isnap,isnap,ifile
       ELSE
          WRITE(filename,'(a,"/postproc_",i3.3,"/sub_tab_",i3.3,".",i1.1)') &
               basedir(1:lnblnk(basedir)),isnap,isnap,ifile
       END IF
    END IF
!
! Read the number of groups
!
    OPEN(UNIT=1,FILE=filename,STATUS='old',ACCESS='direct',RECL=20, &
         ACTION='read',FORM='UNFORMATTED')
    READ(UNIT=1,REC=1)header
    CLOSE(1)
    nfoffile=header(1)
    nidfile=header(2)
    nfoftot=header(3)
    nfiles=header(4)
    nsubfile=header(5)
!
! Allocate storage then read the file
!
    nint=5+(nfoffile*2)+(nsubfile*3)
    nbytes=nint*4
    ALLOCATE(nsphalo(nfoffile),firstsub(nfoffile))
    ALLOCATE(sublen(nsubfile),suboffset(nsubfile),subpar(nsubfile))
    OPEN(UNIT=1,FILE=filename,STATUS='old',ACCESS='direct',RECL=nbytes, &
         ACTION='read')
    READ(UNIT=1,REC=1)header,nsphalo,firstsub,sublen,suboffset,subpar
    CLOSE(1)
!
! Finished!
!
    RETURN
  END SUBROUTINE readsubtab
!
! -----------------------------------------------------------------------
!
  SUBROUTINE readsubids(basedir,isnap,ifile,nfoffile,nidfile,nfoftot, &
       nfiles,groupids)
!
! Read a single group_tab file
!
! Parameters
    USE filepathmod
    IMPLICIT NONE
    CHARACTER(LEN=*), INTENT(IN) :: basedir
    INTEGER, INTENT(IN) :: ifile, isnap
    INTEGER, INTENT(OUT) :: nfoffile,nfoftot,nfiles,nidfile
    INTEGER*8, DIMENSION(:), POINTER :: groupids
! Internal variables
    CHARACTER(LEN=256) :: filename
    INTEGER, DIMENSION(4) :: header
    INTEGER :: nint8, nbytes
    INTEGER :: lnblnk
!
! Check groupids pointer not pointing to anything
!
    IF(ASSOCIATED(groupids)) &
         STOP 'Input pointers already allocated!'
!
! Generate file name
!
    IF(basedir.EQ.'MILLENNIUM')THEN
       filename=filepath(3,isnap,ifile)
    ELSE
       IF(ifile.GT.99)THEN
          WRITE(filename,'(a,"/postproc_",i3.3,"/sub_ids_",i3.3,".",i3.3)') &
               basedir(1:lnblnk(basedir)),isnap,isnap,ifile
       ELSE IF (ifile.GT.9)THEN
          WRITE(filename,'(a,"/postproc_",i3.3,"/sub_ids_",i3.3,".",i2.2)') &
               basedir(1:lnblnk(basedir)),isnap,isnap,ifile
       ELSE
          WRITE(filename,'(a,"/postproc_",i3.3,"/sub_ids_",i3.3,".",i1.1)') &
               basedir(1:lnblnk(basedir)),isnap,isnap,ifile
       END IF
    END IF
!
! Read the number of groups
!
    OPEN(UNIT=1,FILE=filename,STATUS='old',ACCESS='direct',RECL=16, &
         ACTION='read')
    READ(UNIT=1,REC=1)header
    CLOSE(1)
    nfoffile=header(1)
    nidfile=header(2)
    nfoftot=header(3)
    nfiles=header(4)
!
! Allocate storage then read the file
!
    nint8=2+nidfile
    nbytes=nint8*8
    ALLOCATE(groupids(nidfile))
    OPEN(UNIT=1,FILE=filename,STATUS='old',ACCESS='direct',RECL=nbytes, &
         ACTION='read')
    READ(UNIT=1,REC=1)header,groupids
    CLOSE(1)
!
! Finished!
!
    RETURN
  END SUBROUTINE readsubids
!
! -----------------------------------------------------------------------
!
  SUBROUTINE readsubhalos(basedir,isnap,ifile,nfoffile,nsubfile, &
       nidfile,nfoftot,nfiles,halo,subhalo)
!
! Read a single sub_tab file
! 
! This version returns all of the halo and subhalo properties in the arrays
! 'halo' and 'subhalo' which should be declared as
!
! TYPE (subhalotype), POINTER, DIMENSION(:) :: subhalo
! TYPE (halotype), POINTER, DIMENSION(:) :: halo
!
! in the calling program but not allocated, since this routine allocates 
! the arrays to the appropriate size. Halo/subhalo properties can then be 
! accessed using things like
!
! subhalo_x_coordinate = subhalo(isub)%pos(1)
! subhalo_y_coordinate = subhalo(isub)%pos(2)
! subhalo_z_coordinate = subhalo(isub)%pos(3)
! subhalo_half_mass_radius = subhalo(isub)%halfmass
! halo_r200 = halo(ihalo)%rmean200
!
! where isub runs from 1 to nsubfile and ihalo runs from 1 to nfoffile.
! See the type definition at the top of this file for the other (sub)halo
! properties. Note that the 'mostboundid' property is a particle ID so it
! is an INTEGER*8.
!
! Parameters
    USE filepathmod
    IMPLICIT NONE
    CHARACTER(LEN=*), INTENT(IN) :: basedir
    INTEGER, INTENT(IN) :: ifile, isnap
    INTEGER, INTENT(OUT) :: nfoffile,nfoftot,nfiles,nidfile,nsubfile
! Internal variables
    CHARACTER(LEN=256) :: filename
    INTEGER, DIMENSION(5) :: header
    INTEGER :: nint, nbytes
    INTEGER :: lnblnk
    INTEGER :: offset
    REAL, DIMENSION(:,:), ALLOCATABLE :: pos,vel,spin
! Subhalo and halo properties to return
    TYPE (halotype), POINTER, DIMENSION(:) :: halo
    TYPE (subhalotype), POINTER, DIMENSION(:) :: subhalo
!
! Check pointers not already pointing to anything
!
    IF(ASSOCIATED(subhalo).OR.ASSOCIATED(halo)) &
         STOP 'Input pointers already allocated!'
!
! Generate file name
!
    IF(basedir.EQ.'MILLENNIUM')THEN
       filename=filepath(2,isnap,ifile)
    ELSE
       IF(ifile.GT.99)THEN
          WRITE(filename,'(a,"/postproc_",i3.3,"/sub_tab_",i3.3,".",i3.3)') &
               basedir(1:lnblnk(basedir)),isnap,isnap,ifile
       ELSE IF (ifile.GT.9)THEN
          WRITE(filename,'(a,"/postproc_",i3.3,"/sub_tab_",i3.3,".",i2.2)') &
               basedir(1:lnblnk(basedir)),isnap,isnap,ifile
       ELSE
          WRITE(filename,'(a,"/postproc_",i3.3,"/sub_tab_",i3.3,".",i1.1)') &
               basedir(1:lnblnk(basedir)),isnap,isnap,ifile
       END IF
    END IF
!
! Read the number of groups
!
    OPEN(UNIT=1,FILE=filename,STATUS='old',ACCESS='direct',RECL=20, &
         ACTION='read',FORM='UNFORMATTED')
    READ(UNIT=1,REC=1)header
    CLOSE(1)
    nfoffile=header(1)
    nidfile=header(2)
    nfoftot=header(3)
    nfiles=header(4)
    nsubfile=header(5)
!
! Allocate storage then read the file
!
    nint=5+(nfoffile*8)+(nsubfile*17)
    nbytes=nint*4
    ALLOCATE(subhalo(nsubfile),halo(nfoffile))
    ALLOCATE(pos(3,nsubfile),vel(3,nsubfile),spin(3,nsubfile))
    OPEN(UNIT=1,FILE=filename,STATUS='old',ACCESS='direct',RECL=nbytes, &
         ACTION='read')
    READ(UNIT=1,REC=1)header,halo%nsubgroups,halo%firstsubgroup,subhalo%len, &
         subhalo%offset,subhalo%parent, &
         halo%mmean200,halo%rmean200,halo%mcrit200, &
         halo%rcrit200,halo%mtophat200,halo%rtophat200, &
         pos,vel,subhalo%veldisp,subhalo%vmax,spin, &
         subhalo%mostboundid,subhalo%halfmass
    CLOSE(1)
    subhalo(1:nsubfile)%pos(1)=pos(1,1:nsubfile)
    subhalo(1:nsubfile)%pos(2)=pos(2,1:nsubfile)
    subhalo(1:nsubfile)%pos(3)=pos(3,1:nsubfile)
    subhalo(1:nsubfile)%vel(1)=vel(1,1:nsubfile)
    subhalo(1:nsubfile)%vel(2)=vel(2,1:nsubfile)
    subhalo(1:nsubfile)%vel(3)=vel(3,1:nsubfile)
    subhalo(1:nsubfile)%spin(1)=spin(1,1:nsubfile)
    subhalo(1:nsubfile)%spin(2)=spin(2,1:nsubfile)
    subhalo(1:nsubfile)%spin(3)=spin(3,1:nsubfile)
    DEALLOCATE(pos,vel,spin)
!
! Finished!
!
    RETURN
  END SUBROUTINE readsubhalos

END MODULE readgroupsmod
