cmake_minimum_required(VERSION 2.8.0)
project(ch2)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# g2o
#find_package(G2O REQUIRED)
#include_directories(${G2O_INCLUDE_DIRS})

# Eigen
include_directories("/usr/local/include/eigen3")


find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(ceresCurveFitting ceres_gauss_newton_curve_fit.cc)
target_link_libraries(ceresCurveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(handtunedCurveFitting gauss_newton_curve_fit.cc)
target_link_libraries(handtunedCurveFitting ${OpenCV_LIBS})

