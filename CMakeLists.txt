cmake_minimum_required(VERSION 3.8)
project(fast_livo)

# Default to Release build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for clangd

# Common Compile Flags
add_compile_options(-pthread -fexceptions)

# Debug Flags
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  add_compile_options(-O0 -g)
endif()

# Architecture Detection and Optimization
message(STATUS "Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_BUILD_TYPE MATCHES "Release")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64|ARM|AARCH64)")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
      add_compile_options(-O3 -mcpu=native -mtune=native -ffast-math)
      message(STATUS "Using 64-bit ARM optimizations")
    else()
      add_compile_options(-O3 -mcpu=native -mtune=native -mfpu=neon -ffast-math)
      message(STATUS "Using 32-bit ARM optimizations")
    endif()
    add_definitions(-DARM_ARCH)
  else()
    add_compile_options(-O3 -march=native -mtune=native -funroll-loops)
    message(STATUS "Using x86 optimizations")
    add_definitions(-DX86_ARCH)
  endif()
endif()

# Define Project Root
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# Multithreading Detection
include(ProcessorCount)
ProcessorCount(N)
message(STATUS "Processor count: ${N}")
if(N GREATER 4)
  math(EXPR PROC_NUM "4")
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
elseif(N GREATER 1)
  math(EXPR PROC_NUM "${N}")
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

# Find Packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(vikit_common)
find_package(vikit_ros)
find_package(visualization_msgs REQUIRED)

# External Libraries
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenMP)

# Mimalloc
find_package(mimalloc QUIET)
if(mimalloc_FOUND)
  message(STATUS "mimalloc found")
endif()

# Include Directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
)

# OpenMP
if(OpenMP_CXX_FOUND)
  add_compile_options(${OpenMP_CXX_FLAGS})
endif()

# --- Libraries ---

add_library(vio src/vio.cpp src/frame.cpp src/visual_point.cpp)
ament_target_dependencies(vio 
  rclcpp 
  cv_bridge 
  sensor_msgs 
  std_msgs
  tf2_ros
  tf2_geometry_msgs
  vikit_common
  visualization_msgs
)
target_link_libraries(vio ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${Sophus_LIBRARIES})

add_library(lio src/voxel_map.cpp)
ament_target_dependencies(lio 
  rclcpp 
  pcl_conversions 
  sensor_msgs
  tf2
  tf2_geometry_msgs
  visualization_msgs
)
target_link_libraries(lio ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(pre src/preprocess.cpp)
ament_target_dependencies(pre 
  rclcpp 
  pcl_conversions 
  sensor_msgs
  tf2_ros
  livox_ros_driver2
)
target_link_libraries(pre ${PCL_LIBRARIES})

add_library(imu_proc src/IMU_Processing.cpp)
ament_target_dependencies(imu_proc 
  rclcpp
  sensor_msgs 
  geometry_msgs
  nav_msgs
  livox_ros_driver2
  tf2_ros
)
target_link_libraries(imu_proc ${PCL_LIBRARIES} ${Sophus_LIBRARIES})

add_library(laser_mapping src/LIVMapper.cpp)
ament_target_dependencies(laser_mapping 
  rclcpp 
  tf2
  tf2_ros 
  tf2_geometry_msgs
  image_transport 
  nav_msgs
  pcl_conversions
  visualization_msgs
  cv_bridge
  vikit_ros
  livox_ros_driver2
)
target_link_libraries(laser_mapping ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# --- Executable ---

add_executable(fastlivo_mapping src/main.cpp)
ament_target_dependencies(fastlivo_mapping rclcpp)
target_link_libraries(fastlivo_mapping
  laser_mapping
  vio
  lio
  pre
  imu_proc
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
)

if(mimalloc_FOUND)
  target_link_libraries(fastlivo_mapping mimalloc)
endif()

# --- Install Rules (Mandatory for ROS 2) ---

install(TARGETS 
  fastlivo_mapping
  laser_mapping
  vio
  lio
  pre
  imu_proc
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY rviz_cfg
  DESTINATION share/${PROJECT_NAME}
)

ament_package()