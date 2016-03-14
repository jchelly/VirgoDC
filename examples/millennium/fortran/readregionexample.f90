PROGRAM readregionexample
!
! Read all of the particles in a specified volume and write their coordinates
! to an ascii file. The volume is determined by specifying maximum and 
! minimum x,y,z coordinates in comoving Mpc/h.
!
! Method:
!
! 1. Determine the x,y,z coordinates of the hash cells to be read in
!    using the maximum and minimum coordinates given
! 2. Use peano_keys.c to get the hash keys for these cells
! 3. Flag the cells to be read by setting hashmap(hashkey)=true
! 4. Use readregion to read in the particles in these hash cells
!
  USE readregionmod
  IMPLICIT NONE
  CHARACTER(LEN=500) :: basedir, basename, outfile
  INTEGER :: isnap, nhash,nfile,npart
  REAL, POINTER, DIMENSION(:) :: x,y,z,vx,vy,vz
  INTEGER*8, POINTER, DIMENSION(:) :: ids
  LOGICAL, DIMENSION(:), ALLOCATABLE :: hashmap
  REAL :: lbox,xmin,xmax,ymin,ymax,zmin,zmax
  INTEGER :: imin,imax,jmin,jmax,kmin,kmax
  INTEGER :: ncellbox, ihash, hashbits
  INTEGER :: i,j,k
  INTEGER :: peano_hilbert_key

! Get parameters
  WRITE(*,*)'Enter minimum and maximum x coordinates (comoving Mpc/h)' 
  READ(*,*)xmin,xmax
  WRITE(*,*)'Enter minimum and maximum y coordinates (comoving Mpc/h)' 
  READ(*,*)ymin,ymax
  WRITE(*,*)'Enter minimum and maximum z coordinates (comoving Mpc/h)' 
  READ(*,*)zmin,zmax
  WRITE(*,*)'Enter box size'  ! Also in file header, =500Mpc/h for millennium
  READ(*,*)lbox
  WRITE(*,*)'Enter name of directory with snapdir and postproc subdirectories'
  READ(*,'(a)')basedir
  WRITE(*,*)'Enter snapshot basename'
  READ(*,'(a)')basename
  WRITE(*,*)'Enter number of snapshot to read'
  READ(*,*)isnap
  WRITE(*,*)'Enter number of hash cells in simulation' ! =256^3 for millennium
  READ(*,*)nhash
  WRITE(*,*)'Enter number of files per snapshot'
  READ(*,*)nfile   ! =512 for millennium
  WRITE(*,*)'Enter name of output file'
  READ(*,'(a)')outfile

! Find number of hash bits and number of cells across box
  ALLOCATE(hashmap(0:nhash-1))
  hashmap=.FALSE.
  ncellbox=1
  hashbits=0
  DO WHILE(ncellbox*ncellbox*ncellbox.LT.nhash)
     ncellbox=ncellbox*2
     hashbits=hashbits+1
  END DO

  ! Flag cells to read
  imin=MAX(INT((xmin/lbox)*float(ncellbox)),0)
  imax=MIN(INT((xmax/lbox)*float(ncellbox)),ncellbox-1)
  jmin=MAX(INT((ymin/lbox)*float(ncellbox)),0)
  jmax=MIN(INT((ymax/lbox)*float(ncellbox)),ncellbox-1)
  kmin=MAX(INT((zmin/lbox)*float(ncellbox)),0)
  kmax=MIN(INT((zmax/lbox)*float(ncellbox)),ncellbox-1)

  WRITE(*,*)'Flagging cells to read'
  WRITE(*,*)'Cell coordinates in x direction:',imin,' to ',imax
  WRITE(*,*)'Cell coordinates in y direction:',jmin,' to ',jmax
  WRITE(*,*)'Cell coordinates in z direction:',kmin,' to ',kmax

  DO i=imin,imax
     DO j=jmin,jmax
        DO k=kmin,kmax
           ihash=peano_hilbert_key(i,j,k,hashbits)
           hashmap(ihash)=.TRUE.
        END DO
     END DO
  END DO
  WRITE(*,*)'Reading data'

  ! Read the data
  CALL readregion(basedir,basename,isnap,nhash,hashmap,nfile,npart, &
       x,y,z,vx,vy,vz,ids)
  WRITE(*,*)'Read ',npart,' particles'

  ! Discard particles outside specified volume (readregion reads whole
  ! cells so we may have some extra particles)
  j=0
  DO i=1,npart
     IF(x(i).GE.xmin.AND.x(i).LE.xmax.AND.y(i).GE.ymin.AND.y(i).LE.ymax &
          .AND.z(i).GE.zmin.AND.z(i).LE.zmax)THEN
        j=j+1
        x(j)=x(i)
        y(j)=y(i)
        z(j)=z(i)
        vx(j)=vx(i)
        vy(j)=vy(i)
        vz(j)=vz(i)
        ids(j)=ids(i)
     END IF
  END DO
  npart=j
  WRITE(*,*)'Volume contains ',npart,' particles'

  ! Write positions to an ascii file
  WRITE(*,*)'Writing ascii file'
  OPEN(unit=1,file=outfile,status='unknown',form='formatted')
  WRITE(1,'(3e14.6)')(x(i),y(i),z(i),i=1,npart)
  CLOSE(1)

END PROGRAM readregionexample
