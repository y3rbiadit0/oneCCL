#
# Copyright 2026 UXL Foundation. All rights reserved.
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
#

cmake_minimum_required(VERSION 3.20)

project(oneccl_nvshmem_m0 LANGUAGES CXX CUDA)

set(NVSHMEM_M0_CUDA_ARCHITECTURE "80" CACHE STRING
    "CUDA architecture used by the NVSHMEM M0 spike, for example 80 or 90")
set(NVSHMEM_M0_SYCL_TARGET "nvptx64-nvidia-cuda" CACHE STRING
    "DPC++ SYCL target triple used by the NVSHMEM M0 spike")
set(NVSHMEM_ROOT "$ENV{NVSHMEM_HOME}" CACHE PATH
    "NVSHMEM installation prefix used when a CMake package is unavailable")

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fsycl" NVSHMEM_M0_COMPILER_SUPPORTS_FSYCL)
if(NOT NVSHMEM_M0_COMPILER_SUPPORTS_FSYCL)
    message(FATAL_ERROR
        "The NVSHMEM M0 spike requires a DPC++ compiler with -fsycl support. "
        "Set CMAKE_CXX_COMPILER to the DPC++ clang++ or icpx executable.")
endif()

find_package(CUDAToolkit 12 REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(NVSHMEM CONFIG QUIET)

if(TARGET nvshmem::nvshmem_host AND TARGET nvshmem::nvshmem_device)
    set(NVSHMEM_M0_HOST_TARGET nvshmem::nvshmem_host)
    set(NVSHMEM_M0_DEVICE_TARGET nvshmem::nvshmem_device)
    message(STATUS "Using NVSHMEM CMake package targets")
else()
    find_path(NVSHMEM_M0_INCLUDE_DIR
        NAMES nvshmem.h
        HINTS "${NVSHMEM_ROOT}" "$ENV{NVSHMEM_HOME}" "$ENV{NVSHMEM_ROOT}"
        PATH_SUFFIXES include
        NO_DEFAULT_PATH)
    find_library(NVSHMEM_M0_HOST_LIBRARY
        NAMES nvshmem_host
        HINTS "${NVSHMEM_ROOT}" "$ENV{NVSHMEM_HOME}" "$ENV{NVSHMEM_ROOT}"
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH)
    find_library(NVSHMEM_M0_DEVICE_LIBRARY
        NAMES nvshmem_device
        HINTS "${NVSHMEM_ROOT}" "$ENV{NVSHMEM_HOME}" "$ENV{NVSHMEM_ROOT}"
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH)

    if(NOT NVSHMEM_M0_INCLUDE_DIR OR
       NOT EXISTS "${NVSHMEM_M0_INCLUDE_DIR}/nvshmemx.h" OR
       NOT NVSHMEM_M0_HOST_LIBRARY OR
       NOT NVSHMEM_M0_DEVICE_LIBRARY)
        message(FATAL_ERROR
            "NVSHMEM was not found. Set NVSHMEM_DIR for a package installation, "
            "or set NVSHMEM_ROOT/NVSHMEM_HOME to a prefix containing include/ "
            "and lib/{libnvshmem_host.so,libnvshmem_device.a}.")
    endif()

    add_library(oneccl_nvshmem_m0_host SHARED IMPORTED)
    set_target_properties(oneccl_nvshmem_m0_host PROPERTIES
        IMPORTED_LOCATION "${NVSHMEM_M0_HOST_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_M0_INCLUDE_DIR}"
    )

    add_library(oneccl_nvshmem_m0_device STATIC IMPORTED)
    set_target_properties(oneccl_nvshmem_m0_device PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES CUDA
        IMPORTED_LOCATION "${NVSHMEM_M0_DEVICE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_M0_INCLUDE_DIR}"
    )

    set(NVSHMEM_M0_HOST_TARGET oneccl_nvshmem_m0_host)
    set(NVSHMEM_M0_DEVICE_TARGET oneccl_nvshmem_m0_device)
    message(STATUS "Using NVSHMEM libraries discovered under ${NVSHMEM_ROOT}")
endif()

add_executable(oneccl_nvshmem_m0
    nvshmem_m0.cpp
    nvshmem_m0_adapter.cu
)

target_compile_features(oneccl_nvshmem_m0 PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(oneccl_nvshmem_m0 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(oneccl_nvshmem_m0 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fsycl>
    $<$<COMPILE_LANGUAGE:CXX>:-fsycl-targets=${NVSHMEM_M0_SYCL_TARGET}>
    $<$<COMPILE_LANGUAGE:CXX>:-Xsycl-target-backend=${NVSHMEM_M0_SYCL_TARGET}>
    $<$<COMPILE_LANGUAGE:CXX>:--cuda-gpu-arch=sm_${NVSHMEM_M0_CUDA_ARCHITECTURE}>
)

target_link_options(oneccl_nvshmem_m0 PRIVATE
    $<HOST_LINK:-fsycl>
    $<HOST_LINK:-fsycl-targets=${NVSHMEM_M0_SYCL_TARGET}>
    $<HOST_LINK:-Xsycl-target-backend=${NVSHMEM_M0_SYCL_TARGET}>
    $<HOST_LINK:--cuda-gpu-arch=sm_${NVSHMEM_M0_CUDA_ARCHITECTURE}>
)

target_link_libraries(oneccl_nvshmem_m0 PRIVATE
    # NVSHMEM requires applications to link both libraries, including for host-only APIs.
    MPI::MPI_CXX
    ${NVSHMEM_M0_HOST_TARGET}
    ${NVSHMEM_M0_DEVICE_TARGET}
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(oneccl_nvshmem_m0 PROPERTIES
    CUDA_ARCHITECTURES "${NVSHMEM_M0_CUDA_ARCHITECTURE}"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE ON
)

add_executable(oneccl_nvshmem_uid_spike
    nvshmem_uid_spike.cpp
    nvshmem_uid_adapter.cu
)
target_compile_features(oneccl_nvshmem_uid_spike PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(oneccl_nvshmem_uid_spike PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(oneccl_nvshmem_uid_spike PRIVATE
    MPI::MPI_CXX
    ${NVSHMEM_M0_HOST_TARGET}
    CUDA::cudart
    CUDA::cuda_driver
)
set_target_properties(oneccl_nvshmem_uid_spike PROPERTIES
    CUDA_ARCHITECTURES "${NVSHMEM_M0_CUDA_ARCHITECTURE}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE ON
)
