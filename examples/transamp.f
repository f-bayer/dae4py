c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c
c     This file is part of the Test Set for IVP solvers
c     http://www.dm.uniba.it/~testset/
c
c        Transistor Amplifier
c        index 1 DAE of dimension 8
c
c     DISCLAIMER: see
c     http://www.dm.uniba.it/~testset/disclaimer.php
c
c     The most recent version of this source file can be found at
c     http://www.dm.uniba.it/~testset/src/problems/transamp.f
c
c     This is revision
c     $Id: transamp.F,v 1.3 2006/10/25 08:21:22 testset Exp $
c
c-----------------------------------------------------------------------
      integer function pidate()
      pidate = 20060828
      return
      end
c-----------------------------------------------------------------------
      subroutine prob(fullnm,problm,type,
     +                neqn,ndisc,t,
     +                numjac,mljac,mujac,
     +                nummas,mlmas,mumas,
     +                ind)
      character*(*) fullnm, problm, type
      integer neqn,ndisc,mljac,mujac,mlmas,mumas,ind(*)
      double precision t(0:*)
      logical numjac, nummas

      integer i

      fullnm = 'Transistor Amplifier'
      problm = 'transamp'
      type   = 'DAE'
      neqn   = 8
      ndisc  = 0
      t(0)   = 0d0
      t(1)   = 0.2d0
      numjac = .false.
      mljac  = 2
      mujac  = 1
      mlmas  = 1
      mumas  = 1
      do 10 i=1,neqn
         ind(i) = 0
   10 continue

      return
      end
c-----------------------------------------------------------------------
      subroutine init(neqn,t,y,yprime,consis)
      integer neqn
      double precision t,y(neqn),yprime(neqn)
      logical consis

      double precision ub,r1,r2,r3,r5,r6,r7,c2,c4
      parameter (ub=6d0,r1=9000d0,r2=9000d0,r3=9000d0,
     +           r5=9000d0,r6=9000d0,r7=9000d0,c2=2d-6,c4=4d-6)

      y(1) = 0d0
      y(2) = ub/(r2/r1+1d0)
      y(3) = y(2)
      y(4) = ub
      y(5) = ub/(r6/r5+1d0)
      y(6) = y(5)
      y(7) = y(4)
      y(8) = 0d0

      consis = .true.

      yprime(3) = -y(2)/(c2*r3)
      yprime(6) = -y(5)/(c4*r7)
c-----------------------------------------------------------------------
c     the other initial values for yprime are determined numerically
c-----------------------------------------------------------------------
      yprime(1) = 51.338775d0
      yprime(2) = 51.338775d0
      yprime(4) = -24.9757667d0
      yprime(5) = -24.9757667d0
      yprime(7) = -10.00564453d0
      yprime(8) = -10.00564453d0

      return
      end
c-----------------------------------------------------------------------
      subroutine settolerances(neqn,rtol,atol,tolvec)
      integer neqn 
      logical tolvec
      double precision rtol(neqn), atol(neqn)
       
      tolvec  = .false.
      

      return
      end
c-----------------------------------------------------------------------
      subroutine setoutput(neqn,solref,printsolout,
     +                    nindsol,indsol)

      logical solref, printsolout
      integer neqn, nindsol
      integer indsol(neqn)

c the reference solution is available
      solref = .true.  

c output file is required
      printsolout = .true.

c default values if printsolout is .true.
      nindsol = neqn
c only nindsol component of indsol are referenced
      do i=1,nindsol
          indsol(i) = i
      end do  

  

      return
      end

c-----------------------------------------------------------------------
      subroutine feval(neqn,t,y,yprime,f,ierr,rpar,ipar)
      integer neqn,ierr,ipar(*)
      double precision t,y(neqn),yprime(neqn),f(neqn),rpar(*)

      double precision ub,uf,alpha,beta,r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,
     +                 pi,uet,fac1,fac2
      parameter (ub=6d0,uf=0.026d0,alpha=0.99d0,beta=1d-6,
     +           r0=1000d0,r1=9000d0,r2=9000d0,r3=9000d0,
     +           r4=9000d0,r5=9000d0,r6=9000d0,r7=9000d0,
     +           r8=9000d0,r9=9000d0,pi=3.1415926535897931086244d0)

      uet   = 0.1d0*sin(200d0*pi*t)
      
c     prevent overflow
c
c      (double precisione ieee .le. 1d304)
      if ( (y(2)-y(3))/uf.gt.300d0) then
         ierr = -1
         return
      endif
      if ( (y(5)-y(6))/uf.gt.300d0) then
         ierr = -1
         return
      endif
      
      fac1  = beta*(exp((y(2)-y(3))/uf)-1d0)
      fac2  = beta*(exp((y(5)-y(6))/uf)-1d0)

      f(1) = (y(1)-uet)/r0
      f(2) = y(2)/r1+(y(2)-ub)/r2+(1d0-alpha)*fac1
      f(3) = y(3)/r3-fac1
      f(4) = (y(4)-ub)/r4+alpha*fac1
      f(5) = y(5)/r5+(y(5)-ub)/r6+(1d0-alpha)*fac2
      f(6) = y(6)/r7-fac2
      f(7) = (y(7)-ub)/r8+alpha*fac2
      f(8) = y(8)/r9

      return
      end
c-----------------------------------------------------------------------
      subroutine jeval(ldim,neqn,t,y,yprime,dfdy,ierr,rpar,ipar)
      integer ldim,neqn,ierr,ipar(*)
      double precision t,y(neqn),yprime(neqn),dfdy(ldim,neqn),rpar(*)

      integer i
      double precision uf,alpha,beta,r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,
     +                 fac1p,fac2p
      parameter (uf=0.026d0,alpha=0.99d0,beta=1d-6,
     +           r0=1000d0,r1=9000d0,r2=9000d0,r3=9000d0,
     +           r4=9000d0,r5=9000d0,r6=9000d0,r7=9000d0,
     +           r8=9000d0,r9=9000d0)


c     prevent overflow
c
c      (double precisione ieee .le. 1d304)
      if ( (y(2)-y(3))/uf.gt.300d0) then
         ierr = -1
         return
      endif
      if ( (y(5)-y(6))/uf.gt.300d0) then
         ierr = -1
         return
      endif

      fac1p = beta*exp((y(2)-y(3))/uf)/uf
      fac2p = beta*exp((y(5)-y(6))/uf)/uf

      do 10 i=1,8
         dfdy(1,i) = 0d0
         dfdy(3,i) = 0d0
         dfdy(4,i) = 0d0
   10 continue

      dfdy(1,3) = -(1d0-alpha)*fac1p
      dfdy(1,6) = -(1d0-alpha)*fac2p
      dfdy(2,1) = 1d0/r0
      dfdy(2,2) = 1d0/r1+1d0/r2+(1d0-alpha)*fac1p
      dfdy(2,3) = 1d0/r3+fac1p
      dfdy(2,4) = 1d0/r4
      dfdy(2,5) = 1d0/r5+1d0/r6+(1d0-alpha)*fac2p
      dfdy(2,6) = 1d0/r7+fac2p
      dfdy(2,7) = 1d0/r8
      dfdy(2,8) = 1d0/r9
      dfdy(3,2) = -fac1p
      dfdy(3,3) = -alpha*fac1p
      dfdy(3,5) = -fac2p
      dfdy(3,6) = -alpha*fac2p
      dfdy(4,2) = alpha*fac1p
      dfdy(4,5) = alpha*fac2p

      return
      end
c-----------------------------------------------------------------------
      subroutine meval(ldim,neqn,t,y,yprime,dfddy,ierr,rpar,ipar)
      integer ldim,neqn,ierr,ipar(*)
      double precision t,y(neqn),yprime(neqn),dfddy(ldim,neqn),rpar(*)

      integer i
      double precision c1,c2,c3,c4,c5
      parameter (c1=1d-6,c2=2d-6,c3=3d-6,c4=4d-6,c5=5d-6)

      do 10 i=1,neqn
         dfddy(1,i) = 0d0
         dfddy(3,i) = 0d0
   10 continue

      dfddy(1,2) = c1
      dfddy(1,5) = c3
      dfddy(1,8) = c5
      dfddy(2,1) = -c1
      dfddy(2,2) = -c1
      dfddy(2,3) = -c2
      dfddy(2,4) = -c3
      dfddy(2,5) = -c3
      dfddy(2,6) = -c4
      dfddy(2,7) = -c5
      dfddy(2,8) = -c5
      dfddy(3,1) = c1
      dfddy(3,4) = c3
      dfddy(3,7) = c5

      return
      end
c-----------------------------------------------------------------------
      subroutine solut(neqn,t,y)
      integer neqn
      double precision t,y(neqn)
c
c computed on Cray C90 using Cray double precision
c Solving Transistor Amplifier using PSIDE
c
c User input:
c
c give relative error tolerance: 1d-14
c give absolute error tolerance: 1d-14
c
c
c Integration characteristics:
c
c    number of integration steps       16061
c    number of accepted steps          15824
c    number of f evaluations          401944
c    number of Jacobian evaluations      458
c    number of LU decompositions        4884
c
c CPU-time used:                         182.44 sec

      y(  1) = -0.5562145012262709d-002
      y(  2) =  0.3006522471903042d+001
      y(  3) =  0.2849958788608128d+001
      y(  4) =  0.2926422536206241d+001
      y(  5) =  0.2704617865010554d+001
      y(  6) =  0.2761837778393145d+001
      y(  7) =  0.4770927631616772d+001
      y(  8) =  0.1236995868091548d+001

      return
      end
