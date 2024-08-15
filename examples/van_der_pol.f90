PROGRAM VDPOL
!
! PSIDE example: Van der Pol problem
!
! - ODE of dimension 2
!   y' = f
! - formulated as general IDE
!   g = f - y' = 0
! - analytical partial derivative J (full 2x2 matrix)
!   dg/dy = df/dy
! - analytical partial derivative M (band matrix)
!   dg/dy' = -I
!
INTEGER NEQN,NLJ,NUJ,NLM,NUM
LOGICAL JNUM,MNUM
PARAMETER (NEQN=2,NLJ=NEQN,NUJ=NEQN,NLM=0,NUM=0)
! PARAMETER (JNUM=.FALSE., MNUM=.FALSE.)
PARAMETER (JNUM=.TRUE., MNUM=.TRUE.)
INTEGER LRWORK, LIWORK
PARAMETER (LRWORK = 20+27*NEQN+6*NEQN**2, LIWORK = 20+4*NEQN)
INTEGER IND,IWORK(LIWORK),IPAR,IDID
DOUBLE PRECISION Y(NEQN),DY(NEQN),T,TEND,RTOL,ATOL, &
    RWORK(LRWORK),RPAR
EXTERNAL VDPOLG,VDPOLJ,VDPOLM
INTEGER I

!initialize PSIDE
DO 10 I=1,20
IWORK(I) = 0
RWORK(I) = 0D0
10 CONTINUE

! consistent initial values
T = 0D0
Y(1) = 2D0
Y(2) = 0D0
DY(1) = 0D0
DY(2) = -2D0
TEND = 41.5D0

! set scalar tolerances
RTOL = 1D-4
ATOL = 1D-4

WRITE(*, '(1X,A,/)') 'PSIDE example solving Van der Pol problem'

CALL PSIDE(NEQN,Y,DY,VDPOLG, &
    JNUM,NLJ,NUJ,VDPOLJ, &
    MNUM,NLM,NUM,VDPOLM, &
    T,TEND,RTOL,ATOL,IND, &
    LRWORK,RWORK,LIWORK,IWORK, &
    RPAR,IPAR,IDID)

IF (IDID.EQ.1) THEN
WRITE(*,'(1X,A,F5.1)') 'solution at t = ', TEND
    WRITE(*,*)
    DO 20 I=1,NEQN
    WRITE(*,'(4X,''y('',I1,'') ='',E11.3)') I,Y(I)
20 CONTINUE
    WRITE(*,*)
    WRITE(*,'(1X,A,I4)') 'number of steps =', IWORK(15)
    WRITE(*,'(1X,A,I4)') 'number of f-s =', IWORK(11)
    WRITE(*,'(1X,A,I4)') 'number of J-s =', IWORK(12)
    WRITE(*,'(1X,A,I4)') 'number of LU-s =', IWORK(13)
ELSE
    WRITE(*,'(1X,A,I4)') 'PSIDE failed: IDID =', IDID
ENDIF
END

SUBROUTINE VDPOLG(NEQN,T,Y,DY,G,IERR,RPAR,IPAR)
    INTEGER NEQN,IERR,IPAR(*)
    DOUBLE PRECISION T,Y(NEQN),DY(NEQN),G(NEQN),RPAR(*)
    G(1) = Y(2)-DY(1)
    G(2) = 500D0*(1D0-Y(1)*Y(1))*Y(2)-Y(1)-DY(2)
    RETURN
END

SUBROUTINE VDPOLJ(LDJ,NEQN,NLJ,NUJ,T,Y,DY,DGDY,RPAR,IPAR)
    ! INTEGER LDJ,NEQN,NLJ,NUJ,IPAR(*)
    ! DOUBLE PRECISION T,Y(NEQN),DY(NEQN),DGDY(LDJ,NEQN),RPAR(*)
    ! DGDY(1,1) = 0D0
    ! DGDY(1,2) = 1D0
    ! DGDY(2,1) = -1000D0*Y(1)*Y(2)-1D0
    ! DGDY(2,2) = 500D0*(1D0-Y(1)*Y(1))
    ! RETURN
END

SUBROUTINE VDPOLM(LDM,NEQN,NLM,NUM,T,Y,DY,DGDDY,RPAR,IPAR)
    ! INTEGER LDM,NEQN,NLM,NUM,IPAR(*)
    ! DOUBLE PRECISION T,Y(NEQN),DY(NEQN),DGDDY(LDM,NEQN),RPAR(*)
    ! DGDDY(1,1) = -1D0
    ! DGDDY(1,2) = -1D0
    ! RETURN
END