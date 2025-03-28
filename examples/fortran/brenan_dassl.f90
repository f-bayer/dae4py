program brenan
    implicit none
    integer neq, lrwork, liwork, ninfo
    parameter (neq=2, lrwork=40+(5+4)*neq+neq**2, liwork=20+neq, ninfo=15)
    integer iwork(liwork), ipar, idid, info(ninfo)
    double precision y(neq), yp(neq), diff(neq), error, &
                     t, tend, rtol, atol, rwork(lrwork), rpar
    external fun, jac
    integer i
    
    ! consistent initial values
    t = 0.0
    tend = 1000.0
    y(1) = 1.0
    y(2) = 0.0
    yp(1) = -1.0
    yp(2) = 1.0
    
    do i=1,15
        info(i) = 0
    end do

    do i=1,20
        iwork(i) = 0
        rwork(i) = 0.0
    end do

    ! get intermediate results
    info(3) = 1
    ! compute solution until t == t1
    info(4) = 1
    rwork(1) = tend
    
    ! set scalar tolerances
    rtol = 1e-8
    atol = 1e-8
    
    write(*, '(1X,A,/)') "DASSL example solving Brenan's index 1 problem"
    
    ! initialize dassl
    call ddassl(fun, neq, t, y, yp, tend, info, rtol, atol, idid, &
                rwork, lrwork, iwork, liwork, rpar, ipar, jac)

    ! call dassl until tend is reached
    do while (t < tend)
        call ddassl(fun, neq, t, y, yp, tend, info, rtol, atol, idid, &
                    rwork, lrwork, iwork, liwork, rpar, ipar, jac)
    end do

    ! compute error with respect to true solution
    diff(1) = y(1) - (exp(-t) + t * sin(t))
    diff(2) = y(2) - sin(t)
    error = sqrt(diff(1)**2 + diff(2)**2)
    
    write(*,'(1X,A,F7.1)') 'solution at t = ', tend
    write(*,*)
    do i=1,neq
        write(*,'(4X,''y('',I1,'') ='',E11.3)') i, y(i)
    end do
    write(*,*)
    write(*,'(1X,A,E15.5)') 'error = ', error
    write(*,*)
    write(*,'(1X,A,I6)') 'number of steps =', iwork(11)
    write(*,'(1X,A,I6)') 'number of function evaluations =', iwork(12)
    write(*,'(1X,A,I6)') 'number of jacobian evaluations =', iwork(13)
    write(*,'(1X,A,I6)') 'number of error test failures =', iwork(14)
    write(*,'(1X,A,I6)') 'number of convergence test failures =', iwork(15)
    
end program brenan

subroutine fun(t, y, yp, delta, ires, rpar, ipar)
    integer ires, ipar(*)
    double precision t, y(2), yp(2), delta(2), rpar(*)
    delta(1) = yp(1) - t * yp(2) + y(1) - (1.0 + t) * y(2)
    delta(2) = y(2) - sin(t)
    return
end

subroutine jac(t, y, yp, pd, cj, rpar, ipar)
end
