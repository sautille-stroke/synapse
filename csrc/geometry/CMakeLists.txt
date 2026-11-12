###############################################################################
# CMakeLists.txt — synapse native geometry validation kernel
#
# Builds the `_c_geometry` pybind11 extension module from:
#   - src/mesh_checks.cc   (core algorithm implementations)
#   - src/bindings.cc      (pybind11 Python bindings)
#
# Usage (typical pip-driven build via scikit-build-core or manual):
#   cmake -S csrc/geometry -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#
# Prerequisites:
#   - C++20 compiler (Clang 16+, GCC 13+, MSVC 17.6+)
#   - Python 3.10+ development headers
#   - pybind11 (found via find_package; install with `pip install pybind11`)
###############################################################################

cmake_minimum_required(VERSION 3.20)
project(synapse_geometry LANGUAGES CXX)

# ---------- C++20 ----------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------- Dependencies ----------
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# ---------- Sources ----------
set(GEOMETRY_SOURCES
    src/mesh_checks.cc
)

set(BINDING_SOURCES
    src/bindings.cc
)

# ---------- pybind11 extension module ----------
pybind11_add_module(_c_geometry
    ${GEOMETRY_SOURCES}
    ${BINDING_SOURCES}
)

target_include_directories(_c_geometry
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Enable warnings.
if(MSVC)
    target_compile_options(_c_geometry PRIVATE /W4)
else()
    target_compile_options(_c_geometry PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------- Install ----------
# When driven by scikit-build-core this installs into the Python package.
install(TARGETS _c_geometry DESTINATION .)
