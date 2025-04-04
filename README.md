# dae4py - python implementation of solvers for differential algebraic equations (DAE's)

This repository is a collection of examples and solvers for the course [Differential Algebraic Equations I & II](https://www.inm.uni-stuttgart.de/lehre/dae/).

## Fixed step-size implementations

Two fixed step-size implementations are given

* [Implicit Runge-Kutta methods](dae4py/irk/irk.py)
* [Backward differentation formulas (BDF)](dae4py/bdf/bdf.py)

## Variable step-size Radau IIA methods

A variable step-size implementation of the Radau IIA method with odd number of stages is given in [radau.py](dae4py/radau/radau.py).

## Python wrapper for old fortran solvers.

*  Four-stage Radau IIA method pside.f of [de Swart, Lioen and van der Veen](https://archimede.uniba.it/~testset/solvers/pside.php).
* 3 stage Radau IIA method radau.f of [Ernst Hairer](hhttp://www.unige.ch/~hairer/prog/stiff/radau5.f).
* 3, 5 and 7 stage Radau IIA method radau.f of [Ernst Hairer](https://www.unige.ch/~hairer/prog/stiff/radau.f).
* BDF methods of Linda Petzold
    - [ddassl.f](https://www.netlib.org/ode/ddassl.f).
    - [ddaskr.f](https://www.netlib.org/ode/daskr.tgz) (not implemented yet).
    - [ddaspk.f](https://www.netlib.org/ode/daspk.tgz) (not implemented yet).

## Examples

All example discussed in the lecture are implemented here.

## Install

* unix

    ```bash
    python -m venv myvenv
    source myvenv/bin/activate
    python -m pip install .
    ```
* windows
    
    make sure the following programs are installed:
    * [MinGW gcc](https://www.msys2.org/)
    * [Strawberry Perl](https://strawberryperl.com/)
    <!-- * [LAPACK](https://www.netlib.org/lapack/#_lapack_version_3_12_1_2) -->
    ```bash
    python -m venv myvenv
    source myvenv/Scripts/activate
    python -m pip install .
    ```
    ... or go the easy way and use WSL2 with Ubuntu
* MacOS

    tbd