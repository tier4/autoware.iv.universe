cmake_minimum_required(VERSION 3.0.2)
project(ndt_ocl)

# warn performance issue of ndt_ocl which is NOT built in release mode
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} uppercase_CMAKE_BUILD_TYPE)
if (NOT (${uppercase_CMAKE_BUILD_TYPE} STREQUAL "RELEASE"))
  message(WARNING
    "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} may cause a serious performance degradation. Please configure with -DCMAKE_BUILD_TYPE=Release.")
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_ros
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ndt_ocl
  CATKIN_DEPENDS
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(ndt_ocl
  src/ndt_ocl/voxel_grid_covariance_ocl.cpp
  src/ndt_ocl/ndt_ocl.cpp
)

# pcl 1.7 causes a segfault without -O2 (or -O3) flag
target_compile_options(ndt_ocl PUBLIC
  $<$<CONFIG:Debug>:-O2 -g>
)

target_link_libraries(ndt_ocl
  ${catkin_LIBRARIES}
  OpenCL
)

install(TARGETS ndt_ocl
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install project namespaced headers
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
