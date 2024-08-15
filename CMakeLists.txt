cmake_minimum_required(VERSION 3.1)

project(pside VERSION 0.1
        DESCRIPTION "PSIDE solver."
        LANGUAGES Fortran)
enable_language(Fortran)

# # Currently setting the Fortran compiler to use -std=gnu, change this if you
# # want a specific standard
# set(FVERSION "-std=f95")
# set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${FVERSION}")

# define pside library
set(PSIDE_MODS src/pside.f
               src/psided.f
               src/psidea.f
               src/report.f
               CACHE INTERNAL "")
add_library(pside_lib "${PSIDE_MODS}")

# add executable
# add_executable(transamp examples/transamp.f)
# install(TARGETS transamp DESTINATION "bin")
# target_link_libraries(transamp PRIVATE pside_lib)

add_executable(van_der_pol examples/van_der_pol.f90)
install(TARGETS van_der_pol DESTINATION "bin")
target_link_libraries(van_der_pol PRIVATE pside_lib)

add_executable(brenan examples/brenan.f90)
install(TARGETS brenan DESTINATION "bin")
target_link_libraries(brenan PRIVATE pside_lib)