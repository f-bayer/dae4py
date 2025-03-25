! Index 1 DAE found in Chapter 4 of Brenan1996.

! References:
! -----------
! Brenan1996: https://doi.org/10.1137/1.9781611971224.ch4
program amplifier
implicit none

external geval, jeval, meval
! external geval
! integer jeval, meval

integer, parameter :: neqn=2

logical, parameter :: jnum=.true., mnum=.true.
integer, parameter :: nlj=neqn, nuj=neqn
integer, parameter :: nlm=neqn, num=neqn

integer, parameter :: lrwork=20 + 27 * neqn + 6 * neqn**2
real, dimension(lrwork) :: rwork

integer, parameter :: liwork=20 + 4 * neqn
integer, dimension(liwork) :: iwork

integer :: ipar, ind
real :: rpar

integer :: idid

real :: t, tend, atol, rtol
real, dimension(neqn) :: y, yp

integer :: i 

! initial conditions
t = 0.0
tend = 1.0

y(1) = 1.0
y(2) = 0.0

yp(1) = -1.0
yp(2) = 1.0

! tolerances and other solver setup
atol = 1e-6
rtol = 1e-6

do i=1,20
    iwork(i) = 0
    rwork(i) = 0.0
end do

call PSIDE(neqn, y, yp, geval, &
    jnum, nlj, nuj, jeval, &
    mnum, nlm, num, meval, &
    t, tend, rtol, atol, ind, &
    lrwork, rwork, liwork, iwork, &
    rpar, ipar, idid)

end program amplifier

subroutine geval(neqn, t, y, yp, g, ierr, rpar, ipar)
    implicit none
    integer, intent(in) :: neqn, ipar(*)
    integer, intent(out) :: ierr
    real, intent(in) :: t, y(neqn), yp(neqn), rpar(*)
    real, intent(out) :: g(neqn)
    print *, "geval called"
    g(1) = yp(1) - t * yp(2) + y(1) - (1 + t) * y(2)
    g(2) = y(2) - sin(t)
    return

end subroutine geval

subroutine jeval(ldj, neqn, nlj, nuj, t, y, yp, dgdy, rpar, ipar)
    print *, "jeval called"
    return
end subroutine jeval

subroutine meval(ldj, neqn, nlj, nuj, t, y, yp, dgdy, rpar, ipar)
    print *, "meval called"
    return
end subroutine meval
