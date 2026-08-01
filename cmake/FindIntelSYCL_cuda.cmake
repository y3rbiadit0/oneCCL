#===============================================================================
# Copyright 2019 Intel Corporation
# Copyright 2026 Contributors - NVIDIA CUDA/NCCL backend support
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# FindIntelSYCL_cuda.cmake
# This module finds SYCL with CUDA/NVIDIA backend support for use with DPC++
# compiled with NVIDIA GPU support (e.g., from intel/llvm with CUDA backend)

file(GLOB sycl_headers "lib/clang/*/include")

list(APPEND dpcpp_root_hints
            ${DPCPP_ROOT}
            $ENV{DPCPP_ROOT})
set(original_cmake_prefix_path ${CMAKE_PREFIX_PATH})
if(dpcpp_root_hints)
    list(INSERT CMAKE_PREFIX_PATH 0 ${dpcpp_root_hints})
else()
    message("DPCPP_ROOT prefix path hint is not defined")
endif()

message("Searching for SYCL with CUDA backend")

include(CheckCXXCompilerFlag)
include(FindPackageHandleStandardArgs)

# Check if the compiler supports -fsycl
unset(INTEL_SYCL_SUPPORTED CACHE)
check_cxx_compiler_flag("-fsycl" INTEL_SYCL_SUPPORTED)

get_filename_component(INTEL_SYCL_BINARY_DIR ${CMAKE_CXX_COMPILER} PATH)

# Try to find Intel SYCL version.hpp header
find_path(INTEL_SYCL_INCLUDE_DIRS
    NAMES CL/sycl/version.hpp sycl/version.hpp
    PATHS
      ${sycl_root_hints}
      "${INTEL_SYCL_BINARY_DIR}/.."
      "${INTEL_SYCL_BINARY_DIR}/../opt/compiler"
    PATH_SUFFIXES
        include
        include/sycl
        "${sycl_headers}"
    NO_DEFAULT_PATH)

find_library(INTEL_SYCL_LIBRARIES
    NAMES "sycl"
    PATHS
        ${sycl_root_hints}
        "${INTEL_SYCL_BINARY_DIR}/.."
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH)

# Find CUDA toolkit
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA include dir: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA library dir: ${CUDAToolkit_LIBRARY_DIR}")
    set(CUDA_FOUND TRUE)
    set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    set(CUDA_LIBRARIES CUDA::cudart)
else()
    # Fallback: try to find CUDA manually
    find_path(CUDA_INCLUDE_DIRS
        NAMES cuda_runtime.h
        PATHS
            ${CUDA_ROOT}
            $ENV{CUDA_ROOT}
            $ENV{CUDA_HOME}
            $ENV{CUDA_PATH}
            /usr/local/cuda
            /opt/cuda
        PATH_SUFFIXES include
        NO_DEFAULT_PATH)
    
    find_library(CUDA_RUNTIME_LIBRARY
        NAMES cudart
        PATHS
            ${CUDA_ROOT}
            $ENV{CUDA_ROOT}
            $ENV{CUDA_HOME}
            $ENV{CUDA_PATH}
            /usr/local/cuda
            /opt/cuda
        PATH_SUFFIXES lib64 lib
        NO_DEFAULT_PATH)
    
    if(CUDA_INCLUDE_DIRS AND CUDA_RUNTIME_LIBRARY)
        set(CUDA_FOUND TRUE)
        set(CUDA_LIBRARIES ${CUDA_RUNTIME_LIBRARY})
        message(STATUS "CUDA found manually:")
        message(STATUS "  CUDA include dir: ${CUDA_INCLUDE_DIRS}")
        message(STATUS "  CUDA runtime library: ${CUDA_RUNTIME_LIBRARY}")
    else()
        set(CUDA_FOUND FALSE)
        message(WARNING "CUDA toolkit not found. NVIDIA GPU support may be limited.")
    endif()
endif()

# Find NCCL library
find_path(NCCL_INCLUDE_DIRS
    NAMES nccl.h
    PATHS
        ${NCCL_ROOT}
        $ENV{NCCL_ROOT}
        $ENV{NCCL_HOME}
        ${CUDA_ROOT}
        $ENV{CUDA_ROOT}
        $ENV{CUDA_HOME}
        /usr/local/cuda
        /opt/cuda
        /usr
        /usr/local
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)

# Also try default paths
if(NOT NCCL_INCLUDE_DIRS)
    find_path(NCCL_INCLUDE_DIRS NAMES nccl.h)
endif()

find_library(NCCL_LIBRARIES
    NAMES nccl
    PATHS
        ${NCCL_ROOT}
        $ENV{NCCL_ROOT}
        $ENV{NCCL_HOME}
        ${CUDA_ROOT}
        $ENV{CUDA_ROOT}
        $ENV{CUDA_HOME}
        /usr/local/cuda
        /opt/cuda
        /usr
        /usr/local
    PATH_SUFFIXES lib64 lib
    NO_DEFAULT_PATH)

# Also try default paths
if(NOT NCCL_LIBRARIES)
    find_library(NCCL_LIBRARIES NAMES nccl)
endif()

if(NCCL_INCLUDE_DIRS AND NCCL_LIBRARIES)
    set(NCCL_FOUND TRUE CACHE BOOL "NCCL library found" FORCE)
    message(STATUS "NCCL found:")
    message(STATUS "  NCCL include dir: ${NCCL_INCLUDE_DIRS}")
    message(STATUS "  NCCL library: ${NCCL_LIBRARIES}")
else()
    set(NCCL_FOUND FALSE CACHE BOOL "NCCL library found" FORCE)
    message(WARNING "NCCL library not found. Set NCCL_ROOT or NCCL_HOME environment variable.")
endif()

find_package_handle_standard_args(IntelSYCL_cuda
    FOUND_VAR IntelSYCL_cuda_FOUND
    REQUIRED_VARS
        INTEL_SYCL_LIBRARIES
        INTEL_SYCL_INCLUDE_DIRS
        INTEL_SYCL_SUPPORTED)

if(IntelSYCL_cuda_FOUND AND NOT TARGET Intel::SYCL_cuda)
    add_library(Intel::SYCL_cuda UNKNOWN IMPORTED)
    
    # Collect all include directories
    set(SYCL_CUDA_INCLUDE_DIRS "${INTEL_SYCL_INCLUDE_DIRS}")
    if(CUDA_FOUND AND CUDA_INCLUDE_DIRS)
        list(APPEND SYCL_CUDA_INCLUDE_DIRS "${CUDA_INCLUDE_DIRS}")
    endif()
    if(CCL_ENABLE_NCCL AND NCCL_FOUND AND NCCL_INCLUDE_DIRS)
        list(APPEND SYCL_CUDA_INCLUDE_DIRS "${NCCL_INCLUDE_DIRS}")
    endif()

    message(STATUS "IntelSYCL_cuda_FOUND: TRUE")
    message(STATUS "SYCL_CUDA_INCLUDE_DIRS: ${SYCL_CUDA_INCLUDE_DIRS}")

    # Set the SYCL flags for NVIDIA backend
    # -fsycl enables SYCL mode
    # -fsycl-targets=nvptx64-nvidia-cuda targets NVIDIA GPUs
    set(INTEL_SYCL_CUDA_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda")
    
    # Build import libraries list
    set(imp_libs
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${INTEL_SYCL_CUDA_FLAGS}>
        ${COMPUTE_BACKEND_NAME})
    
    # Add CUDA and NCCL libraries if found
    if(CUDA_FOUND)
        if(TARGET CUDA::cudart)
            list(APPEND imp_libs CUDA::cudart)
        elseif(CUDA_LIBRARIES)
            list(APPEND imp_libs ${CUDA_LIBRARIES})
        endif()
    endif()
    
    if(CCL_ENABLE_NCCL AND NCCL_FOUND)
        list(APPEND imp_libs ${NCCL_LIBRARIES})
    endif()

    set_target_properties(Intel::SYCL_cuda PROPERTIES
        INTERFACE_LINK_LIBRARIES "${imp_libs}"
        INTERFACE_INCLUDE_DIRECTORIES "${SYCL_CUDA_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${INTEL_SYCL_LIBRARIES}")
    
    set(INTEL_SYCL_FLAGS "${INTEL_SYCL_CUDA_FLAGS}")
    
    mark_as_advanced(
        INTEL_SYCL_FLAGS
        INTEL_SYCL_LIBRARIES
        INTEL_SYCL_INCLUDE_DIRS
        CUDA_INCLUDE_DIRS
        CUDA_LIBRARIES
        NCCL_INCLUDE_DIRS
        NCCL_LIBRARIES)
endif()
