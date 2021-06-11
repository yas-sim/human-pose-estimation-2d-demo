cmake_minimum_required(VERSION 3.1)
project(pose_extractor)
set(CMAKE_CXX_STANDARD 11)

# Enabling Python
find_package(PythonInterp 3.5 REQUIRED)
find_package(PythonLibs "${PYTHON_VERSION_STRING}" EXACT REQUIRED)
execute_process(
COMMAND "${PYTHON_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
OUTPUT_STRIP_TRAILING_WHITESPACE
RESULT_VARIABLE NUMPY_NOT_FOUND)
if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "NumPy headers not found")
endif()

add_subdirectory(pose_extractor)
