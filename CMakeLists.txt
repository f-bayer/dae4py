cmake_minimum_required(VERSION 3.6)

project(dassl_examples VERSION 0.1
        DESCRIPTION "DASSL examples."
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

# define dassl library
set(DASSL_MODS dae4py/fortran/dassl/src/ddassl.f CACHE INTERNAL "")
add_library(dassl_lib "${DASSL_MODS}")
target_link_libraries(dassl_lib ${LAPACKBLAS_LIBRARIES})

# add executables
# add_executable(van_der_pol examples/fortran/van_der_pol.f90)
# install(TARGETS van_der_pol DESTINATION "bin")
# target_link_libraries(van_der_pol PRIVATE dassl_lib)

add_executable(brenan examples/fortran/brenan_dassl.f90)
install(TARGETS brenan DESTINATION "bin")
target_link_libraries(brenan PRIVATE dassl_lib)

# add_executable(weissinger examples/fortran/weissinger.f90)
# install(TARGETS weissinger DESTINATION "bin")
# target_link_libraries(weissinger PRIVATE dassl_lib)