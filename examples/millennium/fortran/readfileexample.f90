PROGRAM readfileexample

USE readfilemod
IMPLICIT NONE
! Parameters
CHARACTER(LEN=500) :: basedir,basename,outfile
INTEGER :: isnap,ifile
! Particle data
REAL, DIMENSION(:,:), ALLOCATABLE :: pos, vel
INTEGER*8, DIMENSION(:), ALLOCATABLE :: ids
TYPE (headertype) :: header
! Hash table
INTEGER :: firstcell,lastcell, nhash
INTEGER, DIMENSION(:), ALLOCATABLE :: hashtable
INTEGER :: i

! Read parameters
WRITE(*,*)'Enter name of directory with snapdir and postproc subdirectories'
READ(*,'(a)')basedir
WRITE(*,*)'Enter snapshot basename'
READ(*,'(a)')basename
WRITE(*,*)'Enter number of snapshot to read'
READ(*,*)isnap
WRITE(*,*)'Enter number of file to read'
READ(*,*)ifile
WRITE(*,*)'Enter name of output file'
READ(*,'(a)')outfile

! Read the file header
CALL readheader(basedir,basename,isnap,ifile,header)

! Allocate storage for the particles
ALLOCATE(pos(3,header%npart(2)),vel(3,header%npart(2)),ids(header%npart(2)))
ALLOCATE(hashtable(0:header%hashtabsize-1))

! Read the file
CALL readfile(basedir,basename,isnap,ifile,header,firstcell, &
     lastcell,hashtable,pos,vel,ids)

WRITE(*,*)firstcell,lastcell

! Output the particle coordinates
OPEN(unit=1,file=outfile,status='unknown',form='formatted')
WRITE(1,'(1i12,3e14.6)') &
     (ids(i),pos(1,i),pos(2,i),pos(3,i),i=1,header%npart(2),10)
CLOSE(1)

END PROGRAM readfileexample
