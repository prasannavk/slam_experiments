cmake_minimum_required(VERSION 2.8.0)
project(ch3)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


# Eigen
include_directories("/usr/local/include/eigen3")

# Sophus
include_directories("/root/Sophus")

#find_package(Ceres REQUIRED)
#include_directories(${CERES_INCLUDE_DIRS})

add_executable(use_sophus  use_sophus.cc)
target_link_libraries(use_sophus ${OpenCV_LIBS})

