# see https://numpy.org/doc/stable/f2py/buildtools/meson.html
project(
    'dae4py', 
    'c',
)

add_languages('fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

blas = dependency('blas', required: true)
lapack = dependency('lapack', required: true)

_args_blas_lapack = []
lapack_dep = declare_dependency(
  dependencies: [blas, lapack],
  compile_args: _args_blas_lapack
)

subdir('dae4py')