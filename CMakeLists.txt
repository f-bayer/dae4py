# cmake_minimum_required(VERSION 3.6)
cmake_minimum_required(VERSION 3.28)

project(examples VERSION 0.1
        DESCRIPTION "DASSL and IDA examples."
        LANGUAGES Fortran CXX C)
enable_language(Fortran)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build 
        directory) and run CMake from there. You may need to remove CMakeCache.txt.")
endif()

# set a default build type if none was specified
# see https://blog.kitware.com/cmake-and-the-default-build-type/
set(default_build_type "Release")
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
        STRING "Choose the type of build." FORCE)
endif()

find_package(LAPACK REQUIRED)
include_directories(${LAPACK_INCLUDE_DIRS})
include_directories(${BLAS_INCLUDE_DIRS})

if(LAPACK_FOUND AND BLAS_FOUND)
    set(LAPACKBLAS_LIBRARIES ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    message("BLAS FOUND: " ${BLAS_LIBRARIES})
    message("LAPACK FOUND: " ${LAPACK_LIBRARIES})
endif()


# add dependencies via CPM, see https://github.com/TheLartians/CPM.cmake 
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.38.3/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH SHA256=cc155ce02e7945e7b8967ddfaff0b050e958a723ef7aad3766d368940cb15494
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

# SUNDIALS: SUite of Nonlinear and DIfferential/ALgebraic Equation Solvers
message(CHECK_START "Fetching SUNDIALS")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
CPMAddPackage(
    NAME sundials
    URL "https://github.com/LLNL/sundials/releases/download/v7.2.1/sundials-7.2.1.tar.gz"
    # use double precision for floats and 64 bit integers for indices
    OPTIONS "SUNDIALS_PRECISION DOUBLE" "SUNDIALS_INDEX_SIZE 64" "CMAKE_CXX_STANDARD 17" 
        "EXAMPLES_ENABLE_C OFF" "EXAMPLES_INSTALL OFF" "_BUILD_EXAMPLES OFF"
        "BUILD_ARKODE OFF" 
        "BUILD_CVODE OFF"
        "BUILD_CVODES OFF"
        "BUILD_IDA ON"
        "BUILD_IDAS OFF"
        "BUILD_KINSOL OFF"
        "BUILD_SHARED_LIBS OFF"
        "BUILD_STATIC_LIBS ON"
)
list(POP_BACK CMAKE_MESSAGE_INDENT)
message(CHECK_PASS "fetched")

# define dassl library
set(DASSL_MODS dae4py/fortran/dassl/src/ddassl.f CACHE INTERNAL "")
add_library(dassl_lib "${DASSL_MODS}")
target_link_libraries(dassl_lib ${LAPACKBLAS_LIBRARIES})

# build examples
add_executable(brenan_dassl examples/fortran/brenan_dassl.f90)
install(TARGETS brenan_dassl DESTINATION "bin")
target_link_libraries(brenan_dassl PRIVATE dassl_lib)

add_executable(brenan_ida examples/C++/brenan_ida.cpp)
install(TARGETS brenan_ida DESTINATION "bin")
target_link_libraries(brenan_ida PRIVATE sundials_nvecserial sundials_ida)