# see https://numpy.org/doc/stable/f2py/buildtools/meson.html
project(
    'dae4py', 
    'c',
    default_options: [
      'blas=openblas',
      'lapack=openblas'
    ],
)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

add_languages('fortran')

# blas_dep = dependency('blas', required: false)
# lapack_dep = dependency('lapack', required: false)
# if blas_dep.found() and lapack_dep.found()
#   _args_blas_lapack = []
#   blas_lapack_dep = declare_dependency(
#     dependencies: [blas_dep, lapack_dep],
#     compile_args: _args_blas_lapack
#   )
# endif

if true
# TODO: 64-bit BLAS and LAPACK
#
# Note that this works as long as BLAS and LAPACK are detected properly via
# pkg-config. By default we look for OpenBLAS, other libraries can be configured via
# `meson configure -Dblas=blas -Dlapack=lapack` (example to build with Netlib
# BLAS and LAPACK).
# For MKL and for auto-detecting one of multiple libs, we'll need a custom
# dependency in Meson (like is done for scalapack) - see
# https://github.com/mesonbuild/meson/issues/2835
blas_name = get_option('blas')
lapack_name = get_option('lapack')

macOS13_3_or_later = false
if host_machine.system() == 'darwin'
  r = run_command('xcrun', '-sdk', 'macosx', '--show-sdk-version', check: true)
  sdkVersion = r.stdout().strip()

  macOS13_3_or_later = sdkVersion.version_compare('>=13.3')
endif

_args_blas_lapack = []
accelerate_flag = []
generate_blas_wrappers = false
if blas_name == 'accelerate'
  if not macOS13_3_or_later
    error('macOS Accelerate is only supported on macOS >=13.3')
  endif
  if cc.get_id() != 'clang'
    warning('accelerate may not be properly detected with non-native Apple compiler due to https://github.com/mesonbuild/meson/issues/13608')
  endif
  _args_blas_lapack += ['-DACCELERATE_NEW_LAPACK']
  generate_blas_wrappers = true
  accelerate_flag = '-a'
endif

# First try scipy-openblas, and if found don't look for cblas or lapack, we
# know what's inside the scipy-openblas wheels already.
if blas_name == 'openblas' or blas_name == 'auto'
  blas = dependency('scipy-openblas', method: 'pkg-config', required: false)
  if blas.found()
    blas_name = 'scipy-openblas'
    generate_blas_wrappers = true
  endif
endif

# pkg-config uses a lower-case name while CMake uses a capitalized name, so try
# that too to make the fallback detection with CMake work
if blas_name == 'openblas'
  blas = dependency(['openblas', 'OpenBLAS'])
elif blas_name != 'scipy-openblas'  # if so, we found it already
  blas = dependency(blas_name)
endif
blas_dep = declare_dependency(
  dependencies: blas,
  compile_args: _args_blas_lapack
)
if blas_name == 'blas'
  # Netlib BLAS has a separate `libcblas.so` which we use directly in the g77
  # ABI wrappers, so detect it and error out if we cannot find it.
  # In the future, this should be done automatically for:
  #   `dependency('blas', modules: cblas)`
  # see https://github.com/mesonbuild/meson/pull/10921.
  cblas = dependency('cblas')
else
  cblas = []
endif

if 'mkl' in blas.name() or blas.name().to_lower() == 'accelerate' or blas_name == 'scipy-openblas'
  # For these libraries we know that they contain LAPACK, and it's desirable to
  # use that - no need to run the full detection twice.
  lapack = blas
elif lapack_name == 'openblas'
  lapack = dependency(['openblas', 'OpenBLAS'])
else
  lapack = dependency(lapack_name)
endif
lapack_dep = declare_dependency(
  dependencies: lapack,
  compile_args: _args_blas_lapack
)

# dependency_map = {
#   'BLAS': blas,
#   'LAPACK': lapack,
#   'PYBIND11': pybind11_dep,
# }
endif




# scipy_info = run_command('python3', 'scipy_info.py').stdout().strip().split('\n')
# 
# blas_libs = scipy_info[0]
# blas_dirs = scipy_info[1]
# lapack_libs = scipy_info[2]
# lapack_dirs = scipy_info[3]
# 
# blas_lapack_dep = declare_dependency(
#   link_args: [blas_libs, blas_dirs, lapack_libs, lapack_dirs]
# )

#if blas_dep.found() and lapack_dep.found()
#  _args_blas_lapack = []
#  blas_lapack_dep = declare_dependency(
#    dependencies: [blas_dep, lapack_dep],
#    compile_args: _args_blas_lapack
#  )
#endif

subdir('dae4py')