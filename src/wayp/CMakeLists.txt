cmake_minimum_required(VERSION 3.8)
project(wayp)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Use ament_python as the build type
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install the Python modules
ament_python_install_package(${PROJECT_NAME})

# Install launch and resource files
install(
  DIRECTORY launch resource
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
