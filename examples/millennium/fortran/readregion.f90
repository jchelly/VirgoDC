MODULE readregionmod
!
! Seem to need to use module to allow passing of unallocated pointers
! as dummy arguments
!
CONTAINS

SUBROUTINE readregion(basedir,basename,isnap,nhash,hashmap,nfile,npart, &
     x,y,z,vx,vy,vz,ids)
!
! Read in particles for an arbitrary region - region is specified by
! setting hashmap=.true. for required hash cells.
!
! Returns pointers to arrays with the particle data. These pointers
! (x,y,z,vx,vy,vz,ids) must not be allocated before this routine is called.
!
! Set basedir to 'MILLENNIUM' to access the Durham millennium data store - 
! this will automatically generate paths to the required files.
!
USE filepathmod
IMPLICIT NONE
! Filenames and snapshot no.
INTEGER, INTENT(IN) :: isnap, nhash, nfile
CHARACTER(len=*), INTENT(IN) :: basedir, basename
CHARACTER(len=256) :: filename
! Particle data - arrays not yet allocated, passed as pointers
INTEGER, INTENT(OUT) :: npart
REAL, POINTER, DIMENSION(:) :: x,y,z,vx,vy,vz
INTEGER*8, POINTER, DIMENSION(:) :: ids
! Array flagging files and hash cells to read
LOGICAL, DIMENSION(0:nhash-1), INTENT(IN) :: hashmap
! Hash table info
INTEGER, DIMENSION(0:nhash-1) :: hashtable, filetable, npercell
INTEGER, DIMENSION(0:nfile-1) :: lastcell, ninfile
INTEGER :: firstcell
! Loop indices etc
INTEGER :: ifile, ihash, ifcell, ifirst, ilast, nread, ipart
INTEGER :: iopen,i,ix,iy,iz
INTEGER :: lnblnk
INTEGER :: offset
! Set this true for lots of output to the terminal...
LOGICAL, PARAMETER :: verbose = .FALSE. 
INTEGER :: ncells_to_read, ncells_read
!
! Check arrays are not already in use
IF(ASSOCIATED(x))STOP'Array x already allocated in getregion()'
IF(ASSOCIATED(y))STOP'Array y already allocated in getregion()'
IF(ASSOCIATED(z))STOP'Array z already allocated in getregion()'
IF(ASSOCIATED(ids))STOP'Array ids already allocated in getregion()'
IF(ASSOCIATED(vx))STOP'Array vx already allocated in getregion()'
IF(ASSOCIATED(vy))STOP'Array vy already allocated in getregion()'
IF(ASSOCIATED(vz))STOP'Array vz already allocated in getregion()'
!
! Now read in the whole hash table
!
IF(verbose)WRITE(*,*)'Getting hash table from simulation files'
offset=0
DO ifile=0,nfile-1,1
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
   CALL open_gadget(filename)
   CALL readhash(firstcell,lastcell(ifile),hashtable(offset:nhash-1), &
        ninfile(ifile))
   CALL close_gadget()
   offset=lastcell(ifile)+1
   filetable(firstcell:lastcell(ifile))=ifile
END DO

! Loop over all cells and count particles per cell
DO ihash=0,nhash-1,1
   IF(ihash.LT.lastcell(filetable(ihash)))THEN
      npercell(ihash)=hashtable(ihash+1)-hashtable(ihash)
   ELSE
      npercell(ihash)=ninfile(filetable(ihash))-hashtable(ihash)
   END IF
END DO

npart=0
ncells_to_read = 0
DO ihash=0,nhash-1
   IF(hashmap(ihash))then
      npart=npart+npercell(ihash)
      ncells_to_read = ncells_to_read + 1
   ENDIF
   IF(npercell(ihash).LT.0)THEN
      WRITE(*,*)'Negative no. of particles in cell!'
      WRITE(*,*)ihash,npercell(ihash)
   END IF
END DO
IF(verbose)WRITE(*,*) &
     'Reading ',npart,' particles from ',ncells_to_read,' cells'
!
! Allocate storage for particles now we know how many there will be
!
ALLOCATE(x(1:npart))
ALLOCATE(y(1:npart))
ALLOCATE(z(1:npart))
ALLOCATE(vx(1:npart))
ALLOCATE(vy(1:npart))
ALLOCATE(vz(1:npart))
ALLOCATE(ids(1:npart))
!
! Loop over hash cells and try to read particles in largest continuous
! chunks possible
!
ihash=0
ipart=1
iopen=-1
DO WHILE (ihash.LT.nhash)
! Do we need to read this cell?
   IF(hashmap(ihash))THEN
! We want to read this hash cell and any subsequent flagged cells
! in the same file - need to stop if we get to the end of the file
! or the end of the hash table or we get to an un-flagged cell.
      ifcell=ihash
      ifile=filetable(ihash)
      DO WHILE(hashmap(ihash).AND.filetable(ihash).EQ.ifile &
           .AND.ihash.LT.nhash-1)
         ihash=ihash+1
      END DO
      !write(*,*)hashmap(ihash),filetable(ihash),ifile
      IF((.NOT.hashmap(ihash)).OR.filetable(ihash).NE.ifile)then
         ihash=ihash-1
      endif

      if(verbose)then
         ! Summary of cells to read including one extra so
         ! we can see why it stopped
         do i = ifcell, min(ihash+1,nhash-1), 1
            write(*,*)i,hashmap(i),filetable(i)
         end do
      endif

! Now need to read cells ifcell to ihash which are in file number ifile
! Find range of particles we want to read (ifirst=1 would mean start from
! the first particle in the file, ifirst=2 the second etc.)
      ifirst=hashtable(ifcell)+1
      ilast=hashtable(ihash)+npercell(ihash)
      nread=ilast-ifirst+1
      ncells_read = ncells_read + (ihash-ifcell+1)
! Cells may contain no particles (unlikely with sensible parameters)
      IF(nread.GT.0)THEN
! Open file if not already open. May need to close previous file.
         IF(iopen.NE.ifile)THEN
            IF(iopen.GE.0)CALL close_gadget()
            IF(basedir.EQ.'MILLENNIUM')THEN
               filename=filepath(0,isnap,ifile)
            ELSE
               IF(ifile.GT.99)THEN
                  WRITE(filename, &
                       '(a,"/snapdir_",i3.3,"/",a,"_",i3.3,".",i3.3)') &
                       basedir(1:lnblnk(basedir)),isnap, &
                       basename(1:lnblnk(basename)),isnap,ifile
               ELSE IF(ifile.GT.9)THEN
                  WRITE(filename, &
                       '(a,"/snapdir_",i3.3,"/",a,"_",i3.3,".",i2.2)') &
                       basedir(1:lnblnk(basedir)),isnap, &
                       basename(1:lnblnk(basename)),isnap,ifile
               ELSE
                  WRITE(filename, &
                       '(a,"/snapdir_",i3.3,"/",a,"_",i3.3,".",i1.1)') &
                       basedir(1:lnblnk(basedir)),isnap, &
                       basename(1:lnblnk(basename)),isnap,ifile
               END IF
            END IF
            IF(verbose)WRITE(*,*)'Opening file: ',filename(1:lnblnk(filename))
            CALL open_gadget(filename)
            iopen=ifile
         END IF
! Read the particles into array subsections using C routines
         IF(verbose)WRITE(*,*)'Reading particles ',ifirst,' to ',ilast
         IF(verbose)WRITE(*,*)'into array elements ',ipart,' to ', &
              ipart+nread-1,', max = ',size(x),npart
         CALL readpartcoords(ifirst,ilast,x(ipart:ipart+nread-1), &
              y(ipart:ipart+nread-1),z(ipart:ipart+nread-1))
         CALL readpartids(ifirst,ilast,ids(ipart:ipart+nread-1))
         CALL readpartvel(ifirst,ilast,vx(ipart:ipart+nread-1), &
              vy(ipart:ipart+nread-1),vz(ipart:ipart+nread-1))
         ipart=ipart+nread

         if(verbose)write(*,*)'Cells read so far = ',ncells_read

      ELSE
         IF(verbose)WRITE(*,*)'Skipping cell with no particles'
      END IF
   END IF
   ihash=ihash+1
END DO
CALL close_gadget()

END SUBROUTINE
END MODULE
