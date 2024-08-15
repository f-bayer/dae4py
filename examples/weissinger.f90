program weissinger
    implicit none

    integer neqn, nlj, nuj, nlm, num
    logical jnum, mnum
    parameter (neqn=1, nlj=neqn, nuj=neqn, nlm=neqn, num=neqn)
    parameter (jnum=.true., mnum=.true.)
    integer lrwork, liwork
    parameter (lrwork=20+27*neqn+6*neqn**2, liwork=20+4*neqn)
    integer ind, iwork(liwork), ipar, idid
    double precision y(neqn), yp(neqn), diff(neqn), error, &
                     t, tend, rtol, atol, &
                     rwork(lrwork), rpar
    external fun, J, M
    integer i
    
    ! initialize PSIDE
    do i=1,20
        iwork(i) = 0
        rwork(i) = 0.0
    end do
    
    ! consistent initial values
    t = sqrt(0.5)
    tend = 10
    y(1) = sqrt(t**2 + 0.5)
    yp(1) = t / sqrt(t**2 + 0.5)
    
    ! set scalar tolerances
    rtol = 1e-6
    atol = 1e-6
    
    write(*, '(1X,A,/)') "PSIDE example solving Brenan's index 1 problem"
    
    call PSIDE(neqn, y, yp, fun, &
        jnum, nlj, nuj, J, &
        mnum, nlm, num, M, &
        t, tend, rtol, atol, ind, &
        lrwork, rwork, liwork, iwork, &
        rpar, ipar, idid)

    ! compute error with respect to true solution
    diff(1) = y(1) - sqrt(t**2 + 0.5)
    error = sqrt(diff(1)**2)

    if (idid.EQ.1) then
        write(*,'(1X,A,F7.1)') 'solution at t = ', tend
        write(*,*)
        do i=1,neqn
            write(*,'(4X,''y('',I1,'') ='',E11.3)') i, y(i)
        end do
        write(*,*)
        write(*,'(1X,A,I6)') 'number of steps =', iwork(15)
        write(*,'(1X,A,I6)') 'number of f-s =', iwork(11)
        write(*,'(1X,A,I6)') 'number of J-s =', iwork(12)
        write(*,'(1X,A,I6)') 'number of LU-s =', iwork(13)
        write(*,'(1X,A,E15.5)') 'error = ', error
    else
        write(*,'(1X,A,I4)') 'PSIDE failed: idid =', idid
    endif
    
end program weissinger
    
subroutine fun(neqn, t, y, yp, g, ierr, rpar, ipar)
    integer neqn, ierr, ipar(*)
    double precision t, y(neqn), yp(neqn), g(neqn), rpar(*)
    g(1) = t * y(1)**2 * yp(1)**3 - y(1)**3 * yp(1)**2 + t * (t**2 + 1) * yp(1) - t**2 * y(1)
    return
end

subroutine J(ldj, neqn, nlj, nuj, t, y, yp, dgyp, rpar, ipar)
    ! integer ldj, neqn, nlj, nuj, ipar(*)
    ! double precision t, y(neqn), yp(neqn), dgyp(ldj, neqn), rpar(*)
    ! dgyp(1,1) = 0D0
    ! dgyp(1,2) = 1D0
    ! dgyp(2,1) = -1000D0*y(1)*y(2)-1D0
    ! dgyp(2,2) = 500D0*(1D0-y(1)*y(1))
    ! return
end

subroutine M(ldm, neqn, nlm, num, t, y, yp, dgdyp, rpar, ipar)
    ! integer ldm, neqn, nlm, num, ipar(*)
    ! double precision t, y(neqn), yp(neqn), dgdyp(ldm, neqn), rpar(*)
    ! dgdyp(1,1) = -1D0
    ! dgdyp(1,2) = -1D0
    ! return
end