PROGRAM readfofgroups
!
! Read the groups for one millennium file and output the coordinates
! and group membership of all particles in FoF groups.
!
! Method:
!
! 1. Read the group catalog from the group_tab file
! 2. Read the IDs of the particles in each group from the sub_ids file
! 3. Use the hash keys encoded in the unused bits of the particle IDs to
!    determine which hash cells need to be read from the snapshot files
! 4. Use readregion() to read the required hash cells
! 5. For each particle ID in the sub_ids file find the particle from
!    the snapshot file with the same ID
!
! Output:
!
! Ascii table with one line for each particle which is in a FoF group.
! Columns are: 
!
! - Index of FoF halo this particle belongs to
! - x,y,z coordinates of the particle
!
  USE readgroupsmod
  USE readregionmod
  USE readfilemod
  USE sort

  IMPLICIT NONE
! Parameters
  CHARACTER(LEN=500) :: basedir, basename, outfile
  INTEGER :: isnap, nhash,nfile,ifile
! Particle data
  INTEGER :: npart
  REAL, POINTER, DIMENSION(:) :: x,y,z,vx,vy,vz
  INTEGER*8, POINTER, DIMENSION(:) :: ids
  REAL, ALLOCATABLE, DIMENSION(:) :: xgroup,ygroup,zgroup
! Groups
  INTEGER, DIMENSION(:), POINTER :: foflen,foffset
  INTEGER*8, DIMENSION(:), POINTER :: groupids
  INTEGER :: nfoffile,nfoftot,nidfile
! Hash cells to read 
  LOGICAL, DIMENSION(:), ALLOCATABLE :: hashmap
  INTEGER :: i,j
! Sorting
  INTEGER, DIMENSION(:), ALLOCATABLE :: pidx, gidx
  INTEGER*8 :: id
  INTEGER :: ifof
  INTEGER :: nmatched
  INTEGER*8 :: ihash
! File header
  TYPE (headertype) :: header
  REAL :: aexpand 

! Get parameters
  WRITE(*,*)'Enter number of file to read'
  READ(*,*)ifile
  WRITE(*,*)'Enter name of directory with snapdir and postproc subdirectories'
  READ(*,'(a)')basedir
  WRITE(*,*)'Enter snapshot basename'
  READ(*,'(a)')basename
  WRITE(*,*)'Enter number of snapshot to read'
  READ(*,*)isnap
  WRITE(*,*)'Enter name of output file'
  READ(*,'(a)')outfile

  ! Read header from one of the snapshot files
  CALL readheader(basedir,basename,isnap,ifile,header)
  aexpand=REAL(header%time,KIND(aexpand))
  WRITE(*,*)'Expansion factor is ',aexpand
  nhash = header%hashtabsize
  nfile = header%numfiles
  ALLOCATE(hashmap(0:nhash-1))

! Read in the groups from file ifile
  WRITE(*,*)'Reading groups'
  CALL readgrouptab(basedir,isnap,ifile,nfoffile,nidfile,nfoftot, &
       nfile,foflen,foffset)
  CALL readsubids(basedir,isnap,ifile,nfoffile,nidfile,nfoftot, &
       nfile,groupids)

! Flag hash cells containing group particles
  hashmap=.FALSE.
  DO i=1,nidfile
     ihash=ISHFT(groupids(i),-34) ! Hash key is in 30 most significant bits
     hashmap(ihash)=.TRUE.        ! so need to shift right 34 bits
  END DO

! Read the particle data
  WRITE(*,*)'Reading particle data'
  CALL readregion(basedir,basename,isnap,nhash,hashmap,nfile,npart, &
       x,y,z,vx,vy,vz,ids)

! Find coordinates of grouped particles by sorting both list of ids
! Need to discard hash keys encoded in groupids array first
  groupids=ISHFT(ISHFT(groupids,30),-30)
  WRITE(*,*)'Sorting particles'
  ALLOCATE(xgroup(nidfile),ygroup(nidfile),zgroup(nidfile))
  ALLOCATE(gidx(nidfile),pidx(npart))
  CALL sort_index(groupids(1:nidfile),gidx(1:nidfile))
  CALL sort_index(ids(1:npart),pidx(1:npart))

  WRITE(*,*)'Matching particles'
  j=1
  nmatched=0
  DO i=1,nidfile
     id=groupids(gidx(i))
     ! Find particle from simulation file with same ID
     DO WHILE(ids(pidx(j)).LT.id.AND.j.LT.npart)
        j=j+1
     END DO
     IF(id.EQ.ids(pidx(j)))nmatched=nmatched+1
     xgroup(gidx(i))=x(pidx(j))
     ygroup(gidx(i))=y(pidx(j))
     zgroup(gidx(i))=z(pidx(j))
  END DO
  WRITE(*,*)'Matched ',nmatched,' particles out of ',nidfile
  IF(nmatched.LT.nidfile)STOP"Error: Didn't match all particles!"

! Write positions and FoF group membership to an ascii file
! Columns are FoF group ID, then x,y,z coordinates.
  WRITE(*,*)'Outputting results'
  OPEN(unit=1,file=outfile,status='unknown',form='formatted')
  DO ifof=1,nfoffile
     DO i=foffset(ifof)+1,foffset(ifof)+foflen(ifof),1
        WRITE(1,'(i8,3e14.6)') &
             (ifof-1),xgroup(i),ygroup(i),zgroup(i)
     END DO
  END DO
  CLOSE(1)

END PROGRAM readfofgroups

