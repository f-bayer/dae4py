cmake_minimum_required(VERSION 3.6)

project(pside VERSION 0.1
        DESCRIPTION "PSIDE solver."
        LANGUAGES Fortran)
enable_language(Fortran)

find_package(LAPACK REQUIRED)
include_directories(${LAPACK_INCLUDE_DIRS})
include_directories(${BLAS_INCLUDE_DIRS})

if(LAPACK_FOUND AND BLAS_FOUND)
    set(LAPACKBLAS_LIBRARIES ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    message("BLAS FOUND: " ${BLAS_LIBRARIES})
    message("LAPACK FOUND: " ${LAPACK_LIBRARIES})
endif()

# # Currently setting the Fortran compiler to use -std=gnu, change this if you
# # want a specific standard
# set(FVERSION "-std=f95")
# set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${FVERSION}")

# define pside library
set(PSIDE_MODS pside4py/src/pside/pside.f
        #        pside4py/src/pside/psided.f
        #        pside4py/src/pside/psidea.f
        #        pside4py/src/pside/report.f
               CACHE INTERNAL "")
add_library(pside_lib "${PSIDE_MODS}")
target_link_libraries(pside_lib ${LAPACKBLAS_LIBRARIES})

# add executables
add_executable(van_der_pol examples/van_der_pol.f90)
install(TARGETS van_der_pol DESTINATION "bin")
target_link_libraries(van_der_pol PRIVATE pside_lib)

add_executable(brenan examples/brenan.f90)
install(TARGETS brenan DESTINATION "bin")
target_link_libraries(brenan PRIVATE pside_lib)

add_executable(weissinger examples/weissinger.f90)
install(TARGETS weissinger DESTINATION "bin")
target_link_libraries(weissinger PRIVATE pside_lib)