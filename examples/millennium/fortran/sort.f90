module sort
!
! Module to sort arrays of reals, integers or strings
!
! Uses heap sort for large arrays and shell sort for small arrays.
!
  implicit none
  private

  INTEGER, PARAMETER :: real4byte = SELECTED_REAL_KIND(6,37)
  INTEGER, PARAMETER :: real8byte = SELECTED_REAL_KIND(15,307)
  INTEGER, PARAMETER :: int4byte  = SELECTED_INT_KIND(9)
  INTEGER, PARAMETER :: int8byte  = SELECTED_INT_KIND(18)

  ! Callable routines in this module
  public :: sort_index

  ! Array size below which we use a shell sort
  integer, parameter :: nswitch = 100

  ! Subroutine to produce an index to access elements in sorted order
  interface sort_index
     module procedure sort_index_real4
     module procedure sort_index_real8
     module procedure sort_index_integer4
     module procedure sort_index_integer8
     module procedure sort_index_string
     module procedure sort_index_real4_idx8
     module procedure sort_index_real8_idx8
     module procedure sort_index_integer4_idx8
     module procedure sort_index_integer8_idx8
     module procedure sort_index_string_idx8
  end interface

contains

  subroutine sort_index_real4(arr,idx)
!
! Generate an index to access array elements in sorted order.
! This is doing an in place sort on the index array, but comparing the
! referenced element in arr rather than the value stored in idx. 
!
! Note: the array indexes returned in idx start at one, but otherwise we use
!       zero based indexes in this routine.
!
    implicit none
    ! Use zero based array index in this routine
    real(kind=real4byte), dimension(0:), intent(inout) :: arr
    integer(kind=int4byte), dimension(0:), intent(out)   :: idx
    ! Internal variables
    integer(kind=int8byte) :: start, finish
    integer(kind=int8byte) :: root, child
    integer(kind=int8byte) :: n
    integer(kind=int8byte) :: i, j, inc
    ! Temporary variable for swapping values
    ! This needs to be of the same type as the index array.
    integer(kind=int4byte) :: tmp

    ! Check array sizes
    n = size(arr)
    if(size(idx).lt.n)stop'sort_index(): Index array is too small!'

    ! If we have 0 or 1 elements, no sorting required!
    if(n.eq.0)return
    if(n.eq.1)then
       idx(0) = 1
       return
    endif

    ! Initialise index array
    do i = 0, n-1, 1
       idx(i) = i
    end do

    ! If there aren't many elements, use a shell sort
    if(n.lt.nswitch)then
       inc = n / 2
       do while(inc.gt.0)
          do i = inc, n-1, 1
             j = i
             tmp = idx(i)
             do while (j.ge.inc)
                if(arr(idx(j-inc)).le.arr(tmp))exit
                idx(j) = idx(j - inc)
                j = j - inc
             end do
             idx(j) = tmp;
          end do
          if(inc.eq.2)then
             inc = 1
          else 
             inc = floor(inc/2.2)
          end if
       end do
       idx = idx + 1
       return
    endif

    ! Rearrange the data into a heap structure
    start = (n-1)/2
    do while(start.ge.0)
       finish = n - 1
       root   = start
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1))) &
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
       start = start - 1
    end do

    ! Remove elements from the heap in order
    finish = n - 1
    do while(finish.gt.0)
       tmp         = idx(0)
       idx(0)      = idx(finish)
       idx(finish) = tmp
       finish = finish - 1
       root = 0
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1)))&
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
    end do

    ! Return one-based indexes
    idx = idx + 1

    return
  end subroutine sort_index_real4

  subroutine sort_index_real8(arr,idx)
!
! Generate an index to access array elements in sorted order.
! This is doing an in place sort on the index array, but comparing the
! referenced element in arr rather than the value stored in idx. 
!
! Note: the array indexes returned in idx start at one, but otherwise we use
!       zero based indexes in this routine.
!
    implicit none
    ! Use zero based array index in this routine
    real(kind=real8byte), dimension(0:), intent(inout) :: arr
    integer(kind=int4byte), dimension(0:), intent(out)   :: idx
    ! Internal variables
    integer(kind=int8byte) :: start, finish
    integer(kind=int8byte) :: root, child
    integer(kind=int8byte) :: n
    integer(kind=int8byte) :: i, j, inc
    ! Temporary variable for swapping values
    ! This needs to be of the same type as the index array.
    integer(kind=int4byte) :: tmp

    ! Check array sizes
    n = size(arr)
    if(size(idx).lt.n)stop'sort_index(): Index array is too small!'

    ! If we have 0 or 1 elements, no sorting required!
    if(n.eq.0)return
    if(n.eq.1)then
       idx(0) = 1
       return
    endif

    ! Initialise index array
    do i = 0, n-1, 1
       idx(i) = i
    end do

    ! If there aren't many elements, use a shell sort
    if(n.lt.nswitch)then
       inc = n / 2
       do while(inc.gt.0)
          do i = inc, n-1, 1
             j = i
             tmp = idx(i)
             do while(j.ge.inc)
                if(arr(idx(j-inc)).le.arr(tmp))exit
                idx(j) = idx(j - inc)
                j = j - inc
             end do
             idx(j) = tmp;
          end do
          if(inc.eq.2)then
             inc = 1
          else 
             inc = floor(inc/2.2)
          end if
       end do
       idx = idx + 1
       return
    endif

    ! Rearrange the data into a heap structure
    start = (n-1)/2
    do while(start.ge.0)
       finish = n - 1
       root   = start
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1))) &
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
       start = start - 1
    end do

    ! Remove elements from the heap in order
    finish = n - 1
    do while(finish.gt.0)
       tmp         = idx(0)
       idx(0)      = idx(finish)
       idx(finish) = tmp
       finish = finish - 1
       root = 0
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1)))&
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
    end do

    ! Return one-based indexes
    idx = idx + 1

    return
  end subroutine sort_index_real8

  subroutine sort_index_integer4(arr,idx)
!
! Generate an index to access array elements in sorted order.
! This is doing an in place sort on the index array, but comparing the
! referenced element in arr rather than the value stored in idx. 
!
! Note: the array indexes returned in idx start at one, but otherwise we use
!       zero based indexes in this routine.
!
    implicit none
    ! Use zero based array index in this routine
    integer(kind=int4byte), dimension(0:), intent(inout) :: arr
    integer(kind=int4byte), dimension(0:), intent(out)   :: idx
    ! Internal variables
    integer(kind=int8byte) :: start, finish
    integer(kind=int8byte) :: root, child
    integer(kind=int8byte) :: n
    integer(kind=int8byte) :: i, j, inc
    ! Temporary variable for swapping values
    ! This needs to be of the same type as the index array.
    integer(kind=int4byte) :: tmp

    ! Check array sizes
    n = size(arr)
    if(size(idx).lt.n)stop'sort_index(): Index array is too small!'

    ! If we have 0 or 1 elements, no sorting required!
    if(n.eq.0)return
    if(n.eq.1)then
       idx(0) = 1
       return
    endif

    ! Initialise index array
    do i = 0, n-1, 1
       idx(i) = i
    end do

    ! If there aren't many elements, use a shell sort
    if(n.lt.nswitch)then
       inc = n / 2
       do while(inc.gt.0)
          do i = inc, n-1, 1
             j = i
             tmp = idx(i)
             do while (j.ge.inc)
                if(arr(idx(j-inc)).le.arr(tmp))exit
                idx(j) = idx(j - inc)
                j = j - inc
             end do
             idx(j) = tmp;
          end do
          if(inc.eq.2)then
             inc = 1
          else 
             inc = floor(inc/2.2)
          end if
       end do
       idx = idx + 1
       return
    endif

    ! Rearrange the data into a heap structure
    start = (n-1)/2
    do while(start.ge.0)
       finish = n - 1
       root   = start
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1))) &
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
       start = start - 1
    end do

    ! Remove elements from the heap in order
    finish = n - 1
    do while(finish.gt.0)
       tmp         = idx(0)
       idx(0)      = idx(finish)
       idx(finish) = tmp
       finish = finish - 1
       root = 0
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1)))&
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
    end do

    ! Return one-based indexes
    idx = idx + 1

    return
  end subroutine sort_index_integer4

  subroutine sort_index_integer8(arr,idx)
!
! Generate an index to access array elements in sorted order.
! This is doing an in place sort on the index array, but comparing the
! referenced element in arr rather than the value stored in idx. 
!
! Note: the array indexes returned in idx start at one, but otherwise we use
!       zero based indexes in this routine.
!
    implicit none
    ! Use zero based array index in this routine
    integer(kind=int8byte), dimension(0:), intent(inout) :: arr
    integer(kind=int4byte), dimension(0:), intent(out)   :: idx
    ! Internal variables
    integer(kind=int8byte) :: start, finish
    integer(kind=int8byte) :: root, child
    integer(kind=int8byte) :: n
    integer(kind=int8byte) :: i, j, inc
    ! Temporary variable for swapping values
    ! This needs to be of the same type as the index array.
    integer(kind=int4byte) :: tmp

    ! Check array sizes
    n = size(arr)
    if(size(idx).lt.n)stop'sort_index(): Index array is too small!'

    ! If we have 0 or 1 elements, no sorting required!
    if(n.eq.0)return
    if(n.eq.1)then
       idx(0) = 1
       return
    endif

    ! Initialise index array
    do i = 0, n-1, 1
       idx(i) = i
    end do

    ! If there aren't many elements, use a shell sort
    if(n.lt.nswitch)then
       inc = n / 2
       do while(inc.gt.0)
          do i = inc, n-1, 1
             j = i
             tmp = idx(i)
             do while(j.ge.inc)
                if(arr(idx(j-inc)).le.arr(tmp))exit
                idx(j) = idx(j - inc)
                j = j - inc
             end do
             idx(j) = tmp;
          end do
          if(inc.eq.2)then
             inc = 1
          else 
             inc = floor(inc/2.2)
          end if
       end do
       idx = idx + 1
       return
    endif

    ! Rearrange the data into a heap structure
    start = (n-1)/2
    do while(start.ge.0)
       finish = n - 1
       root   = start
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1))) &
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
       start = start - 1
    end do

    ! Remove elements from the heap in order
    finish = n - 1
    do while(finish.gt.0)
       tmp         = idx(0)
       idx(0)      = idx(finish)
       idx(finish) = tmp
       finish = finish - 1
       root = 0
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1)))&
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
    end do

    ! Return one-based indexes
    idx = idx + 1

    return
  end subroutine sort_index_integer8
  




  subroutine sort_index_string(arr,idx)
!
! Generate an index to access array elements in sorted order.
! This is doing an in place sort on the index array, but comparing the
! referenced element in arr rather than the value stored in idx. 
!
! Note: the array indexes returned in idx start at one, but otherwise we use
!       zero based indexes in this routine.
!
    implicit none
    ! Use zero based array index in this routine
    character(len=*),       dimension(0:), intent(inout) :: arr
    integer(kind=int4byte), dimension(0:), intent(out)   :: idx
    ! Internal variables
    integer(kind=int8byte) :: start, finish
    integer(kind=int8byte) :: root, child
    integer(kind=int8byte) :: n
    integer(kind=int8byte) :: i, j, inc
    ! Temporary variable for swapping values
    ! This needs to be of the same type as the index array.
    integer(kind=int4byte) :: tmp

    ! Check array sizes
    n = size(arr)
    if(size(idx).lt.n)stop'sort_index(): Index array is too small!'

    ! If we have 0 or 1 elements, no sorting required!
    if(n.eq.0)return
    if(n.eq.1)then
       idx(0) = 1
       return
    endif

    ! Initialise index array
    do i = 0, n-1, 1
       idx(i) = i
    end do

    ! If there aren't many elements, use a shell sort
    if(n.lt.nswitch)then
       inc = n / 2
       do while(inc.gt.0)
          do i = inc, n-1, 1
             j = i
             tmp = idx(i)
             do while(j.ge.inc)
                if(arr(idx(j-inc)).le.arr(tmp))exit
                idx(j) = idx(j - inc)
                j = j - inc
             end do
             idx(j) = tmp;
          end do
          if(inc.eq.2)then
             inc = 1
          else 
             inc = floor(inc/2.2)
          end if
       end do
       idx = idx + 1
       return
    endif

    ! Rearrange the data into a heap structure
    start = (n-1)/2
    do while(start.ge.0)
       finish = n - 1
       root   = start
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1))) &
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
       start = start - 1
    end do

    ! Remove elements from the heap in order
    finish = n - 1
    do while(finish.gt.0)
       tmp         = idx(0)
       idx(0)      = idx(finish)
       idx(finish) = tmp
       finish = finish - 1
       root = 0
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1)))&
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
    end do

    ! Return one-based indexes
    idx = idx + 1

    return
  end subroutine sort_index_string


!
! Sort index for 8 byte indexes
!

  subroutine sort_index_real4_idx8(arr,idx)
!
! Generate an index to access array elements in sorted order.
! This is doing an in place sort on the index array, but comparing the
! referenced element in arr rather than the value stored in idx. 
!
! Note: the array indexes returned in idx start at one, but otherwise we use
!       zero based indexes in this routine.
!
    implicit none
    ! Use zero based array index in this routine
    real(kind=real4byte), dimension(0:), intent(inout) :: arr
    integer(kind=int8byte), dimension(0:), intent(out)   :: idx
    ! Internal variables
    integer(kind=int8byte) :: start, finish
    integer(kind=int8byte) :: root, child
    integer(kind=int8byte) :: n
    integer(kind=int8byte) :: i, j, inc
    ! Temporary variable for swapping values
    ! This needs to be of the same type as the index array.
    integer(kind=int8byte) :: tmp

    ! Check array sizes
    n = size(arr)
    if(size(idx).lt.n)stop'sort_index(): Index array is too small!'

    ! If we have 0 or 1 elements, no sorting required!
    if(n.eq.0)return
    if(n.eq.1)then
       idx(0) = 1
       return
    endif

    ! Initialise index array
    do i = 0, n-1, 1
       idx(i) = i
    end do

    ! If there aren't many elements, use a shell sort
    if(n.lt.nswitch)then
       inc = n / 2
       do while(inc.gt.0)
          do i = inc, n-1, 1
             j = i
             tmp = idx(i)
             do while (j.ge.inc)
                if(arr(idx(j-inc)).le.arr(tmp))exit
                idx(j) = idx(j - inc)
                j = j - inc
             end do
             idx(j) = tmp;
          end do
          if(inc.eq.2)then
             inc = 1
          else 
             inc = floor(inc/2.2)
          end if
       end do
       idx = idx + 1
       return
    endif

    ! Rearrange the data into a heap structure
    start = (n-1)/2
    do while(start.ge.0)
       finish = n - 1
       root   = start
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1))) &
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
       start = start - 1
    end do

    ! Remove elements from the heap in order
    finish = n - 1
    do while(finish.gt.0)
       tmp         = idx(0)
       idx(0)      = idx(finish)
       idx(finish) = tmp
       finish = finish - 1
       root = 0
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1)))&
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
    end do

    ! Return one-based indexes
    idx = idx + 1

    return
  end subroutine sort_index_real4_idx8

  subroutine sort_index_real8_idx8(arr,idx)
!
! Generate an index to access array elements in sorted order.
! This is doing an in place sort on the index array, but comparing the
! referenced element in arr rather than the value stored in idx. 
!
! Note: the array indexes returned in idx start at one, but otherwise we use
!       zero based indexes in this routine.
!
    implicit none
    ! Use zero based array index in this routine
    real(kind=real8byte), dimension(0:), intent(inout) :: arr
    integer(kind=int8byte), dimension(0:), intent(out)   :: idx
    ! Internal variables
    integer(kind=int8byte) :: start, finish
    integer(kind=int8byte) :: root, child
    integer(kind=int8byte) :: n
    integer(kind=int8byte) :: i, j, inc
    ! Temporary variable for swapping values
    ! This needs to be of the same type as the index array.
    integer(kind=int8byte) :: tmp

    ! Check array sizes
    n = size(arr)
    if(size(idx).lt.n)stop'sort_index(): Index array is too small!'

    ! If we have 0 or 1 elements, no sorting required!
    if(n.eq.0)return
    if(n.eq.1)then
       idx(0) = 1
       return
    endif

    ! Initialise index array
    do i = 0, n-1, 1
       idx(i) = i
    end do

    ! If there aren't many elements, use a shell sort
    if(n.lt.nswitch)then
       inc = n / 2
       do while(inc.gt.0)
          do i = inc, n-1, 1
             j = i
             tmp = idx(i)
             do while(j.ge.inc)
                if(arr(idx(j-inc)).le.arr(tmp))exit
                idx(j) = idx(j - inc)
                j = j - inc
             end do
             idx(j) = tmp;
          end do
          if(inc.eq.2)then
             inc = 1
          else 
             inc = floor(inc/2.2)
          end if
       end do
       idx = idx + 1
       return
    endif

    ! Rearrange the data into a heap structure
    start = (n-1)/2
    do while(start.ge.0)
       finish = n - 1
       root   = start
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1))) &
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
       start = start - 1
    end do

    ! Remove elements from the heap in order
    finish = n - 1
    do while(finish.gt.0)
       tmp         = idx(0)
       idx(0)      = idx(finish)
       idx(finish) = tmp
       finish = finish - 1
       root = 0
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1)))&
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
    end do

    ! Return one-based indexes
    idx = idx + 1

    return
  end subroutine sort_index_real8_idx8

  subroutine sort_index_integer4_idx8(arr,idx)
!
! Generate an index to access array elements in sorted order.
! This is doing an in place sort on the index array, but comparing the
! referenced element in arr rather than the value stored in idx. 
!
! Note: the array indexes returned in idx start at one, but otherwise we use
!       zero based indexes in this routine.
!
    implicit none
    ! Use zero based array index in this routine
    integer(kind=int4byte), dimension(0:), intent(inout) :: arr
    integer(kind=int8byte), dimension(0:), intent(out)   :: idx
    ! Internal variables
    integer(kind=int8byte) :: start, finish
    integer(kind=int8byte) :: root, child
    integer(kind=int8byte) :: n
    integer(kind=int8byte) :: i, j, inc
    ! Temporary variable for swapping values
    ! This needs to be of the same type as the index array.
    integer(kind=int8byte) :: tmp

    ! Check array sizes
    n = size(arr)
    if(size(idx).lt.n)stop'sort_index(): Index array is too small!'

    ! If we have 0 or 1 elements, no sorting required!
    if(n.eq.0)return
    if(n.eq.1)then
       idx(0) = 1
       return
    endif

    ! Initialise index array
    do i = 0, n-1, 1
       idx(i) = i
    end do

    ! If there aren't many elements, use a shell sort
    if(n.lt.nswitch)then
       inc = n / 2
       do while(inc.gt.0)
          do i = inc, n-1, 1
             j = i
             tmp = idx(i)
             do while (j.ge.inc)
                if(arr(idx(j-inc)).le.arr(tmp))exit
                idx(j) = idx(j - inc)
                j = j - inc
             end do
             idx(j) = tmp;
          end do
          if(inc.eq.2)then
             inc = 1
          else 
             inc = floor(inc/2.2)
          end if
       end do
       idx = idx + 1
       return
    endif

    ! Rearrange the data into a heap structure
    start = (n-1)/2
    do while(start.ge.0)
       finish = n - 1
       root   = start
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1))) &
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
       start = start - 1
    end do

    ! Remove elements from the heap in order
    finish = n - 1
    do while(finish.gt.0)
       tmp         = idx(0)
       idx(0)      = idx(finish)
       idx(finish) = tmp
       finish = finish - 1
       root = 0
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1)))&
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
    end do

    ! Return one-based indexes
    idx = idx + 1

    return
  end subroutine sort_index_integer4_idx8

  subroutine sort_index_integer8_idx8(arr,idx)
!
! Generate an index to access array elements in sorted order.
! This is doing an in place sort on the index array, but comparing the
! referenced element in arr rather than the value stored in idx. 
!
! Note: the array indexes returned in idx start at one, but otherwise we use
!       zero based indexes in this routine.
!
    implicit none
    ! Use zero based array index in this routine
    integer(kind=int8byte), dimension(0:), intent(inout) :: arr
    integer(kind=int8byte), dimension(0:), intent(out)   :: idx
    ! Internal variables
    integer(kind=int8byte) :: start, finish
    integer(kind=int8byte) :: root, child
    integer(kind=int8byte) :: n
    integer(kind=int8byte) :: i, j, inc
    ! Temporary variable for swapping values
    ! This needs to be of the same type as the index array.
    integer(kind=int8byte) :: tmp

    ! Check array sizes
    n = size(arr)
    if(size(idx).lt.n)stop'sort_index(): Index array is too small!'

    ! If we have 0 or 1 elements, no sorting required!
    if(n.eq.0)return
    if(n.eq.1)then
       idx(0) = 1
       return
    endif

    ! Initialise index array
    do i = 0, n-1, 1
       idx(i) = i
    end do

    ! If there aren't many elements, use a shell sort
    if(n.lt.nswitch)then
       inc = n / 2
       do while(inc.gt.0)
          do i = inc, n-1, 1
             j = i
             tmp = idx(i)
             do while(j.ge.inc)
                if(arr(idx(j-inc)).le.arr(tmp))exit
                idx(j) = idx(j - inc)
                j = j - inc
             end do
             idx(j) = tmp;
          end do
          if(inc.eq.2)then
             inc = 1
          else 
             inc = floor(inc/2.2)
          end if
       end do
       idx = idx + 1
       return
    endif

    ! Rearrange the data into a heap structure
    start = (n-1)/2
    do while(start.ge.0)
       finish = n - 1
       root   = start
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1))) &
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
       start = start - 1
    end do

    ! Remove elements from the heap in order
    finish = n - 1
    do while(finish.gt.0)
       tmp         = idx(0)
       idx(0)      = idx(finish)
       idx(finish) = tmp
       finish = finish - 1
       root = 0
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1)))&
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
    end do

    ! Return one-based indexes
    idx = idx + 1

    return
  end subroutine sort_index_integer8_idx8
  




  subroutine sort_index_string_idx8(arr,idx)
!
! Generate an index to access array elements in sorted order.
! This is doing an in place sort on the index array, but comparing the
! referenced element in arr rather than the value stored in idx. 
!
! Note: the array indexes returned in idx start at one, but otherwise we use
!       zero based indexes in this routine.
!
    implicit none
    ! Use zero based array index in this routine
    character(len=*),       dimension(0:), intent(inout) :: arr
    integer(kind=int8byte), dimension(0:), intent(out)   :: idx
    ! Internal variables
    integer(kind=int8byte) :: start, finish
    integer(kind=int8byte) :: root, child
    integer(kind=int8byte) :: n
    integer(kind=int8byte) :: i, j, inc
    ! Temporary variable for swapping values
    ! This needs to be of the same type as the index array.
    integer(kind=int8byte) :: tmp

    ! Check array sizes
    n = size(arr)
    if(size(idx).lt.n)stop'sort_index(): Index array is too small!'

    ! If we have 0 or 1 elements, no sorting required!
    if(n.eq.0)return
    if(n.eq.1)then
       idx(0) = 1
       return
    endif

    ! Initialise index array
    do i = 0, n-1, 1
       idx(i) = i
    end do

    ! If there aren't many elements, use a shell sort
    if(n.lt.nswitch)then
       inc = n / 2
       do while(inc.gt.0)
          do i = inc, n-1, 1
             j = i
             tmp = idx(i)
             do while(j.ge.inc)
                if(arr(idx(j-inc)).le.arr(tmp))exit
                idx(j) = idx(j - inc)
                j = j - inc
             end do
             idx(j) = tmp;
          end do
          if(inc.eq.2)then
             inc = 1
          else 
             inc = floor(inc/2.2)
          end if
       end do
       idx = idx + 1
       return
    endif

    ! Rearrange the data into a heap structure
    start = (n-1)/2
    do while(start.ge.0)
       finish = n - 1
       root   = start
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1))) &
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
       start = start - 1
    end do

    ! Remove elements from the heap in order
    finish = n - 1
    do while(finish.gt.0)
       tmp         = idx(0)
       idx(0)      = idx(finish)
       idx(finish) = tmp
       finish = finish - 1
       root = 0
       do while(2*root+1.le.finish) 
          child = root * 2 + 1
          if(child.lt.finish)then
             if(arr(idx(child)).lt.arr(idx(child+1)))&
                  child = child + 1
          endif
          if(arr(idx(root)).lt.arr(idx(child))) then
             tmp        = idx(root)
             idx(root)  = idx(child)
             idx(child) = tmp
             root = child
          else
             exit
          endif
       end do
    end do

    ! Return one-based indexes
    idx = idx + 1

    return
  end subroutine sort_index_string_idx8

end module sort


