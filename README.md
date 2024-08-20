# dae4py

Python wrapper for old fortran/C DAE solvers.

1. Four-stage Radau IIA method pside.f of [de Swart, Lioen and van der Veen](https://archimede.uniba.it/~testset/solvers/pside.php).
2. 3, 5 and 7 stage Radau IIA method radau.f of [Ernst Hairer](https://www.unige.ch/~hairer/prog/stiff/radau.f) (not implemented yet).
3. BDF methods of Linda Petzold
    - [ddassl.f](https://www.netlib.org/ode/ddassl.f) (not implemented yet).
    - [ddaskr.f](https://www.netlib.org/ode/daskr.tgz) (not implemented yet).
    - [ddaspk.f](https://www.netlib.org/ode/daspk.tgz) (not implemented yet).