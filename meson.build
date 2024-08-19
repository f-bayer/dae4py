# see https://numpy.org/doc/stable/f2py/buildtools/meson.html
project(
    'pside4py', 
    'c',
)

add_languages('fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

blas = dependency('blas')
lapack = dependency('lapack')

_args_blas_lapack = []
lapack_dep = declare_dependency(
  dependencies: lapack,
  compile_args: _args_blas_lapack
)

subdir('pside')