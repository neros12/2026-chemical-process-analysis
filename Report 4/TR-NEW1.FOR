c -----------------------------------------
c transient solutions for newtonian fluids
c in the isothermal process.
c -----------------------------------------

      implicit double precision (a-h,o-z)
      parameter (nx=500,tol=1.d-10)
      common /area / a(0:nx),aold(0:nx)
      common /vel  / v(0:nx),vold(0:nx),vtemp(0:nx)
      common /dmat / cc(nx,3),rr(nx)
      common /const/ coeff,dx,dt,drawr,dstbns

      print *,'draw ratio and disturbance :'
      read(*,*) v(nx), dstbns
      print *,'dimensionless time :'
      read(*,*) n_time

      dx = 1.d0/dfloat(nx)
      dt = dx/dfloat(5)
      coeff = 8.d0*dx/dt

      jj = 0
      iter1 = 0
      tstep = dfloat(jj)*dt

      call steady
      drawr1 = v(nx)*dstbns
      print *,'new draw ratio:',v(nx)

      open (unit=1,file='18a.dat',access='append')
      write(*,*) jj,tstep,a(nx)
      write(1,*) tstep,a(nx)
      close(1)

      ntime = n_time/dt
      print *,'ntime:',ntime

c --------------------------------------------------------------------
      do 100 jj=1,ntime
      tstep = dt*dfloat(jj)

      do 110 i=0,nx
      vold(i) = v(i)
      aold(i) = a(i)
110   continue

      v(nx) = drawr1

      iter = 0
10    iter = iter + 1

      do 120 i=0,nx
      vtemp(i) = v(i)
120   continue

      call cmat_a
      call cmat_v

      do 150 i=0,nx
      if(dabs(v(i)-vtemp(i)).gt.tol) goto 10
150   continue

      if(tstep.le.10.d0) then
         open (unit=1,file='18a.dat',access='append')
         if(mod(jj,100).eq.0) write(*,*) jj,tstep,a(nx)
         if(mod(jj,5).eq.0)   write(1,*) tstep,a(nx)
         close(1)

      else if(tstep.gt.10.d0.and.tstep.le.20.d0) then
         open (unit=2,file='18b.dat',access='append')
         if(mod(jj,100).eq.0) write(*,*) jj,tstep,a(nx)
         if(mod(jj,5).eq.0)   write(2,*) tstep,a(nx)
         close(2)

      else if(tstep.gt.20.d0.and.tstep.le.30.d0) then
         open (unit=3,file='18c.dat',access='append')
         if(mod(jj,100).eq.0) write(*,*) jj,tstep,a(nx)
         if(mod(jj,5).eq.0)   write(3,*) tstep,a(nx)
         close(3)
      endif

100   continue

      stop
      end

c --------------------------------------------------------------------
      subroutine steady
      implicit double precision (a-h,o-z)
      parameter (nx=500,tol=1.d-10)
      common /area / a(0:nx),aold(0:nx)
      common /vel  / v(0:nx),vold(0:nx),vtemp(0:nx)
      common /dmat / cc(nx,3),rr(nx)
      common /const/ coeff,dx,dt,drawr,dstbns

      do 10 i=0,nx
      v(i) = v(nx)**(dx*dfloat(i))
      a(i) = 1.d0/v(i)
10    continue

      return
      end

c --------------------------------------------------------------------
      subroutine cmat_a
      implicit double precision (a-h,o-z)
      parameter (nx=500,tol=1.d-10)
      common /area / a(0:nx),aold(0:nx)
      common /vel  / v(0:nx),vold(0:nx),vtemp(0:nx)
      common /dmat / cc(nx,3),rr(nx)
      common /const/ coeff,dx,dt,drawr,dstbns

      do 10 i=1,nx-1
      cc(i,1) = -vold(i-1) - v(i-1)
      cc(i,2) = coeff
      cc(i,3) = vold(i+1) + v(i+1)
      rr(i)   = (vold(i-1)+v(i-1))*aold(i-1) + coeff*aold(i)
     &        - (vold(i+1)+v(i+1))*aold(i+1)
10    continue
      rr(1)   = (vold(0)+v(0))*aold(0) + coeff*aold(1)
     &        - (vold(2)+v(2))*aold(2) + (vold(0)+v(0))
      cc(1,1) = 0.d0
      cc(nx,1)= -4.d0*vold(nx-1)-4.d0*v(nx-1)+coeff
      cc(nx,2)= 4.d0*vold(nx)+4.d0*v(nx)+coeff
      cc(nx,3)= 0.d0
      rr(nx)  = (coeff+4.d0*vold(nx-1)+4.d0*v(nx-1))*aold(nx-1)
     &        + (coeff-4.d0*vold(nx)-4.d0*v(nx))*aold(nx)
      cc(nx,1)= cc(nx,1)/cc(nx,2)
      rr(nx)  = rr(nx)/cc(nx,2)

      call diagonal(a,nx)
      return
      end

c --------------------------------------------------------------------
      subroutine cmat_v
      implicit double precision (a-h,o-z)
      parameter (nx=500,tol=1.d-10)
      common /area / a(0:nx),aold(0:nx)
      common /vel  / v(0:nx),vold(0:nx),vtemp(0:nx)
      common /dmat / cc(nx,3),rr(nx)
      common /const/ coeff,dx,dt,drawr,dstbns

      do 10 i=1,nx-1
      cc(i,1) = a(i-1)+4.d0*a(i)-a(i+1)
      cc(i,2) = -8.d0*a(i)
      cc(i,3) = -a(i-1)+4.d0*a(i)+a(i+1)
      rr(i)   = 0.d0
10    continue
      cc(1,1) = 0.d0
      cc(nx-1,3) = 0.d0
      rr(1)   = -a(0)-4.d0*a(1)+a(2)
      rr(nx-1)= (a(nx-2)-4.d0*a(nx-1)-a(nx))*v(nx)
      cc(nx-1,1) = cc(nx-1,1)/cc(nx-1,2)
      rr(nx-1) = rr(nx-1)/cc(nx-1,2)

      call diagonal(v,nx-1)

      return
      end

c --------------------------------------------------------------------
      subroutine diagonal(x,n)
      implicit double precision (a-h,o-z)
      parameter (nx=500,tol=1.d-10)
      common /area / a(0:nx),aold(0:nx)
      common /vel  / v(0:nx),vold(0:nx),vtemp(0:nx)
      common /dmat / cc(nx,3),rr(nx)
      common /const/ coeff,dx,dt,drawr,dstbns

      double precision x(0:nx)

      do 10 i=2,n
      m=n-i+2
      temp=1.d0/(cc(m-1,2)-cc(m,1)*cc(m-1,3))
      cc(m-1,1) = cc(m-1,1)*temp
      rr(m-1) = (rr(m-1)-cc(m-1,3)*rr(m))*temp
10    continue
      x(1) = rr(1)

      do 20 i=2,n
      x(i) = rr(i) - cc(i,1)*x(i-1)
20    continue

      return
      end

                                                                      





