# Version Requirement for CMake
cmake_minimum_required(VERSION 3.23)
# Project Name
project(NickTorch)
# Set C++ Standard to C++11
set(CMAKE_CXX_STANDARD 11)
# Ensure all warnings are on
add_compile_options(-Wall -Wextra -pedantic -Werror)

# The header files for the project
set(HEADER_FILES include/matrix.h include/graph.h include/type.h)
# Add the source files
set(SRC_FILES src/matrix.cpp src/type.cpp src/graph.cpp src/main.cpp)
# Add the main executable to the build script
add_executable(NickTorch ${SRC_FILES})
# Include the header files in the build process
target_include_directories(NickTorch PRIVATE include)
