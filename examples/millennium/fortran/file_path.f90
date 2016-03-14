MODULE filepathmod

CONTAINS

  FUNCTION filepath(itype, isnap, ifile) 
    !
    !     Return path to file number ifile of snapshot isnap. itype determines
    !     file type. Path is returned as a 256 character string.
    !
    !     0 - snapshot file
    !     1 - group_tab file
    !     2 - sub_tab file
    !     3 - sub_ids file
    !     4 - ids file
    !
    IMPLICIT NONE
    CHARACTER(LEN=256) :: filepath 
    CHARACTER(LEN=256) :: dir, filename
    INTEGER :: itype, isnap, ifile, idest
    CHARACTER(LEN=4) :: fileno
    INTEGER :: lnblnk

    !     Find which /data/milli?? directory the file is in
    idest = mod(ifile+3*(63-isnap),14)

    !     Generate the number to go on the end of the filename
    IF(ifile.LT.10)THEN
       WRITE(fileno,'(".",1i1.1)')ifile
    ELSE IF (ifile.LT.100)THEN
       WRITE(fileno,'(".",1i2.2)')ifile
    ELSE
       WRITE(fileno,'(".",1i3.3)')ifile
    END IF

    !     Get the directory path and filename
    IF(itype.EQ.0)THEN
       WRITE(dir,'("/data/milli",i2.2,"/d",i2.2,"/snapshot/")') &
            idest,isnap
       WRITE(filename,'("snap_millennium_",i3.3,a)')isnap,fileno
    ELSE IF(itype.EQ.1)THEN
       WRITE(dir,'("/data/milli",i2.2,"/d",i2.2,"/group_tab/")') &
            idest,isnap
       WRITE(filename,'("group_tab_",i3.3,a)')isnap,fileno
    ELSE IF(itype.EQ.2)THEN
       WRITE(dir,'("/data/milli",i2.2,"/d",i2.2,"/sub_tab/")') &
            idest,isnap
       WRITE(filename,'("sub_tab_",i3.3,a)')isnap,fileno
    ELSE IF(itype.EQ.3)THEN
       WRITE(dir,'("/data/milli",i2.2,"/d",i2.2,"/sub_ids/")') &
            idest,isnap
       WRITE(filename,'("sub_ids_",i3.3,a)')isnap,fileno
    ELSE IF(itype.EQ.4)THEN
       WRITE(dir,'("/data/milli",i2.2,"/d",i2.2,"/group_ids/")') &
            idest,isnap
       WRITE(filename,'("group_ids_",i3.3,a)')isnap,fileno
    ELSE
       STOP 'itype out of range!'
    END IF

    filepath=dir(1:lnblnk(dir))//filename(1:lnblnk(filename))
    RETURN

  END FUNCTION filepath

END MODULE filepathmod
