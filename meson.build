# build project with:
# meson setup builddir
# meson compile -C builddir
# cd builddir && python -c "import numpy as np; import fibby; a = np.zeros(9); fibby.fib(a); print (a)" && cd ..
project('f2py_examples', 'c',
  version : '0.1',
  license: 'BSD-3',
  meson_version: '>=0.64.0',
  default_options : ['warning_level=2'],
)

add_languages('fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

# List all Fortran source files
fortran_sources = [
  'src/pside.f', 
  'src/psidea.f', 
  'src/report.f',
  'src/psided.f', 
]

# Generate the custom target for the wrapper code
f2py_sources = custom_target('f2pymodule.c',
  input : fortran_sources,  # List all Fortran sources here
  output : ['f2pymodule.c', 'f2py-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'f2py', '--lower']
)

inc_np = include_directories(incdir_numpy, incdir_f2py)

# Define the extension module
py.extension_module('pside',
  fortran_sources + [f2py_sources],  # Include all Fortran sources and the generated C wrapper
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : py_dep,
  install : true
)

# pside_lib = static_library('pside_lib',
#   pside_source,
#   # fortran_args: fortran_ignore_warnings,
#   override_options: ['b_lto=false'],
#   gnu_symbol_visibility: 'hidden',
# )

# # pside_source = custom_target('psidemodule.c',
# #   input : ['src/pside.f', 'src/psided.f', 'src/psidea.f', 'src/report.f'],  # .f so no F90 wrappers
# #   output : ['psidemodule.c', 'pside-f2pywrappers.f'],
# #   command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'pside', '--lower']
# # )

# pside_module = custom_target('psidemodule.c',
#   input : pside_lib,  # .f so no F90 wrappers
#   output : ['psidemodule.c', 'pside-f2pywrappers.f'],
#   command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'pside', '--lower']
# )

# inc_np = include_directories(incdir_numpy, incdir_f2py)

# # py.extension_module('pside',
# #   # ['src/pside.f', pside_source],
# #   ['src/pside.f', pside_source],
# #   incdir_f2py / 'fortranobject.c',
# #   include_directories: inc_np,
# #   dependencies : py_dep,
# #   install : true
# # )

# py.extension_module('pside',
#   '_odepackmodule.c',
#   # fortran_args: fortran_ignore_warnings,
#   link_with: [pside_lib],
#   # link_args: version_link_args,
#   # dependencies: [lapack_dep, np_dep],
#   include_directories: inc_np,
#   install: true,
#   link_language: 'fortran',
#   # subdir: 'scipy/integrate'
# )