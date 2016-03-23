cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# This file modifies the default compiler flags for certain compilers
SET(CMAKE_USER_MAKE_RULES_OVERRIDE ${CMAKE_SOURCE_DIR}/CompilerFlags.txt)

project(hdf5_wrapper)
enable_language(C)
enable_language(Fortran)

#
# Set location to create Fortran module and library
# ("make install" copies to /include and /lib under ${CMAKE_INSTALL_PREFIX})
#
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

#
# Override cmake's default behaviour of stripping library
# paths from executables
#
SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#
# Find HDF5
#
find_package(HDF5)
include_directories(${HDF5_INCLUDE_DIRS})
find_library(HDF5_C_LIBS hdf5 ${HDF5_LIBRARY_DIRS})
find_library(HDF5_F90_LIBS hdf5_fortran ${HDF5_LIBRARY_DIRS})

#
# Determine how to call C from Fortran
#
include(FortranCInterface)
FortranCInterface_HEADER(FC.h MACRO_NAMESPACE "FC_")
include_directories(${CMAKE_BINARY_DIR})

#
# Build library
#
add_subdirectory(src)

#
# Build test programs
#
add_subdirectory(test)

#
# Packaging stuff
# This allows you to generate a source tar file with "make package_source"
#
set(CPACK_SOURCE_GENERATOR TGZ)
set(CPACK_SOURCE_PACKAGE_FILE_NAME "hdf5_wrapper" CACHE INTERNAL "tarball basename")
set(CPACK_SOURCE_IGNORE_FILES
"~$"
"\\\\.bzrignore$"
"\\\\/.bzr/"
)
include(CPack)
