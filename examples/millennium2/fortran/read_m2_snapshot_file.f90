program read_mill2_snapshot
!
! Read pos, vel, id of DM particles in a Millennium-2 snapshot file
! and write them to standard output
!
  implicit none
  ! Input file name
  character(len=500) :: fname
  ! Particle data
  real*4,    dimension(:,:), allocatable :: pos
  real*4,    dimension(:,:), allocatable :: vel
  integer*8, dimension(:),   allocatable :: id
  integer :: i
  ! Variables for computing particle number
  integer*8 :: nptot(0:5)
  integer*8 :: nptot_hw(0:5)

  ! Gadget file header
  type headertype
     integer*4, dimension(0:5) :: npart
     real*8,    dimension(0:5) :: mass
     real*8 :: time
     real*8 :: redshift
     integer*4 :: flag_sfr
     integer*4 :: flag_feedback
     integer*4, dimension(0:5) :: nparttotal
     integer*4 :: flag_cooling
     integer*4 :: numfiles
     real*8 :: boxsize
     real*8 :: omega0
     real*8 :: omegalambda
     real*8 :: hubbleparam
     integer*4 :: flag_stellarage
     integer*4 :: flag_metals
     integer*4, dimension(0:5)  :: nparttotal_hw
     character, dimension(64) :: unused
  end type headertype
  type (headertype) :: header

  ! Get name of file to read
  call get_command_argument(1, fname)

  ! Open the file
  open(unit=1, file=fname, status="old", form="unformatted")

  ! Read header
  read(1)header
  
  !
  ! Calculate total number of particles in full snapshot.
  ! This is complicated by the fact that nparttotal and nparttotal_hw
  ! are unsigned in the file but Fortran doesn't have unsigned integers.
  !
  nptot    = header%nparttotal     ! Copy value to 8 byte ints
  nptot_hw = header%nparttotal_hw
  do i= 0, 5, 1
     ! Undo overflow by adding 2**32 if negative
     if(nptot(i).lt.0)   nptot(i)    = nptot(i)    + int(2, kind(nptot))**32
     if(nptot_hw(i).lt.0)nptot_hw(i) = nptot_hw(i) + int(2, kind(nptot))**32
  end do
  nptot = nptot + ishft(nptot_hw, 32)
  write(*,*)"Total number of particles = ", nptot(1)
  write(*,*)"Box size = ", header%boxsize

  ! Read particle data
  allocate(pos(3, header%npart(1)))
  read(1)pos
  allocate(vel(3, header%npart(1)))
  read(1)vel
  allocate(id(header%npart(1)))
  read(1)id

  ! Close file
  close(1)

  ! Write out results
  write(*,*)"x,y,z,vx,vy,vz,id"
  do i = 1, header%npart(1), 1
     write(*,'(6e14.6,1i16)')pos(1:3,i), vel(1:3,i), id(i)
  end do

end program read_mill2_snapshot
