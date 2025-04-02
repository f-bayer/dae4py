# see https://numpy.org/doc/stable/f2py/buildtools/meson.html
project(
    'dae4py', 
    'c',
)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

fortran_dep = dependency('fortran', required: false)
if fortran_dep.found()
  add_languages('fortran')
endif

blas_dep = dependency('blas', required: false)
lapack_dep = dependency('lapack', required: false)

if blas_dep.found() and lapack_dep.found()
  _args_blas_lapack = []
  blas_lapack_dep = declare_dependency(
    dependencies: [blas_dep, lapack_dep],
    compile_args: _args_blas_lapack
  )
endif

subdir('dae4py')