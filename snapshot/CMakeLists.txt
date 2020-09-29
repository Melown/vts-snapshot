# bootstrap build system
cmake_minimum_required(VERSION 3.10)
project(vts-snapshot)

# by default, this is OSS build (can be overridden by customer machinery)
set(BUILDSYS_DEFAULT_CUSTOMER_NAME oss)
set(BUILDSYS_CPP_STANDARD 17)
include(buildsys/cmake/buildsys.cmake)

# vts browser
add_subdirectory(src/browser EXCLUDE_FROM_ALL)
add_library(VtsBrowser ALIAS vts-browser)
add_library(VtsRenderer ALIAS vts-renderer)
include_directories(src/browser/src/vts-libbrowser/externals)
include_directories(src/browser/src/vts-libbrowser/include)
include_directories(src/browser/src/vts-librenderer/include)

# third-party dependencies
find_package(OpenCV REQUIRED)
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(TIFF REQUIRED)
find_package(EXIF REQUIRED)
find_package(Boost 1.46 REQUIRED
    COMPONENTS thread program_options filesystem system
    date_time serialization regex chrono iostreams)
link_directories(${Boost_LIBRARY_DIRS})
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# parallelism wouldn't hurt
enable_OpenMP()

# finds OpenGL & co.
enable_OpenGL(OpenGL EGL)

# dependencies
add_subdirectory(src/service)
add_subdirectory(src/glsupport)
add_subdirectory(src/vtsoffscreen)

add_subdirectory(src/snapshot)
