# dae4py - python implementation of solvers for differential algebraic equations (DAE's)

tbd

## Fixed step-size implementations

tbd

## Variable step-size Radau IIA methods

tbd

## Python wrapper for old fortran/C solvers.

*  Four-stage Radau IIA method pside.f of [de Swart, Lioen and van der Veen](https://archimede.uniba.it/~testset/solvers/pside.php).
* 3 stage Radau IIA method radau.f of [Ernst Hairer](hhttp://www.unige.ch/~hairer/prog/stiff/radau5.f).
* 3, 5 and 7 stage Radau IIA method radau.f of [Ernst Hairer](https://www.unige.ch/~hairer/prog/stiff/radau.f).
* BDF methods of Linda Petzold
    - [ddassl.f](https://www.netlib.org/ode/ddassl.f).
    - [ddaskr.f](https://www.netlib.org/ode/daskr.tgz) (not implemented yet).
    - [ddaspk.f](https://www.netlib.org/ode/daspk.tgz) (not implemented yet).

## Examples

tbd

## Install

* unix

    ```bash
    python -m venv myvenv
    source myvenv/bin/activate
    python -m pip install .
    ```
* windows

    tbd

* MacOS

    tbd