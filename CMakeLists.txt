cmake_minimum_required(VERSION 3.15)

project(pose_extractor)

set(CMAKE_BUILD_TYPE "Release")

find_package(PythonInterp 3.5 REQUIRED)
find_package(PythonLibs "${PYTHON_VERSION_STRING}" EXACT REQUIRED)
add_subdirectory(pose_extractor)
