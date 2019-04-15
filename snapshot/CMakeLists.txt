# bootstrap build system
cmake_minimum_required(VERSION 3.10)
project(vts-snapshot)

# by default, this is OSS build (can be overrided by customer machinery)
set(BUILDSYS_DEFAULT_CUSTOMER_NAME oss)
# enforce C++17
set(BUILDSYS_CPP_STANDARD 17)
include(buildsys/cmake/buildsys.cmake)

# allow VTS libs to reside locally
list(APPEND CMAKE_PREFIX_PATH $ENV{HOME}/install)

# parallelism wouldn't hurt
enable_OpenMP()

# some boost
find_package(Boost 1.58 REQUIRED
  COMPONENTS thread program_options filesystem system date_time
             serialization regex chrono iostreams)
link_directories(${Boost_LIBRARY_DIRS})
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

# opencv
find_package(OpenCV REQUIRED)

# image formats
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(TIFF REQUIRED)
find_package(EXIF REQUIRED)

find_package(GDAL REQUIRED)
include_directories(SYSTEM ${GDAL_INCLUDE_DIR})
find_package(Proj REQUIRED)
include_directories(${PROJ_INCLUDE_DIR})
find_package(GeographicLib REQUIRED)
include_directories(${GEOGRAPHICLIB_INCLUDE_DIR})
find_package(JsonCPP REQUIRED)
include_directories(${JSONCPP_INCLUDE_DIRS})

# vts browser library
find_package(VtsBrowser REQUIRED)
include_directories(SYSTEM ${VtsBrowser_INCLUDE_DIR})

# vts renderer library
find_package(VtsRenderer REQUIRED)
include_directories(SYSTEM ${VtsRenderer_INCLUDE_DIR})

# finds OpenGL & co.
enable_OpenGL(OpenGL EGL)

# dependencies
add_subdirectory(src/dbglog)
add_subdirectory(src/utility)
add_subdirectory(src/service)
add_subdirectory(src/math)
add_subdirectory(src/imgproc)
add_subdirectory(src/geometry)
add_subdirectory(src/jsoncpp)
add_subdirectory(src/geo)
add_subdirectory(src/glsupport)
add_subdirectory(src/vtsoffscreen)

add_subdirectory(src/snapshot)
