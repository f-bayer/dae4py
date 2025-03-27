1) File name convention:  filename-nnn.dat
   nnn = streamwise nodal (cell center) location
  
   nnn = 181  --->  x/h = -3.0
   nnn = 360  --->  x/h = 4.0
   nnn = 411  --->  x/h = 6.0
   nnn = 513  --->  x/h = 10.0
   nnn = 641  --->  x/h = 15.0
   nnn = 744  --->  x/h = 19.0

2) Files "x-nnn.dat" contain U, V, u'(rms), v'(rms), w'(rms), u'v'.  
   File header explains the data columns.

   All quantities are nomalized to U0, where U0 is the mean inlet 
   free stream velocity.

3) Files "r**-nnn.dat" contain Reynolds stress budgets.

   rs11-nnn.dat  --->  Streamwise component (u'u')
   rs22-nnn.dat  --->  Vertical component (v'v')
   rs33-nnn.dat  --->  Spanwise component (w'w')
   rs12-nnn.dat  --->  Reynolds shear stress component (u'v')
   rskk-nnn.dat  --->  Turbulent kinetic energy

   All quantities are nomalized to U0^3/h.

4) File "stat-info.dat" contains additional information at each streamwise 
   location (u_tau, Cf, etc...).
