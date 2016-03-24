program read_hdf5_snapshot
!
! Read pos, vel, id of DM particles in a HDF5 snapshot file
! and write them to standard output
!
  use hdf5_wrapper
  implicit none
  ! Input file name
  character(len=500) :: fname
  ! File handle
  integer :: ifile
  ! Particle data
  integer*4 :: npfile(0:5)
  real*4,    dimension(:,:), allocatable :: pos
  real*4,    dimension(:,:), allocatable :: vel
  integer*8, dimension(:),   allocatable :: id
  integer :: i

  ! Get name of file to read
  call get_command_argument(1, fname)

  ! Open the file
  call hdf5_open_file(ifile, fname, readonly=.true.)
  call hdf5_read_attribute(ifile, "Header/NumPart_ThisFile", npfile)

  ! Read positions
  allocate(pos(3, npfile(1)))
  call hdf5_read_data(ifile, "PartType1/Coordinates", pos)

  ! Read velocities
  allocate(vel(3, npfile(1)))
  call hdf5_read_data(ifile, "PartType1/Velocities", vel)
 
  ! Read IDs
  allocate(id(npfile(1)))
  call hdf5_read_data(ifile, "PartType1/ParticleIDs", id)

  call hdf5_close_file(ifile)

  ! Write out results
  write(*,*)"x,y,z,vx,vy,vz,id"
  do i = 1, npfile(1), 1
     write(*,'(6e14.6,1i16)')pos(1:3,i), vel(1:3,i), id(i)
  end do

end program read_hdf5_snapshot
