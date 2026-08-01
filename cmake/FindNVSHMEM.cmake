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

include(FindPackageHandleStandardArgs)

set(NVSHMEM_ROOT "$ENV{NVSHMEM_HOME}" CACHE PATH "NVSHMEM installation prefix")

find_package(NVSHMEM CONFIG QUIET NO_MODULE)

if(TARGET nvshmem::nvshmem_host AND TARGET nvshmem::nvshmem_device)
    set(NVSHMEM_HOST_TARGET nvshmem::nvshmem_host)
    set(NVSHMEM_DEVICE_TARGET nvshmem::nvshmem_device)
else()
    find_path(NVSHMEM_INCLUDE_DIR
        NAMES nvshmem.h
        HINTS "${NVSHMEM_ROOT}" "$ENV{NVSHMEM_HOME}" "$ENV{NVSHMEM_ROOT}"
        PATH_SUFFIXES include)
    find_library(NVSHMEM_HOST_LIBRARY
        NAMES nvshmem_host
        HINTS "${NVSHMEM_ROOT}" "$ENV{NVSHMEM_HOME}" "$ENV{NVSHMEM_ROOT}"
        PATH_SUFFIXES lib lib64)
    find_library(NVSHMEM_DEVICE_LIBRARY
        NAMES nvshmem_device
        HINTS "${NVSHMEM_ROOT}" "$ENV{NVSHMEM_HOME}" "$ENV{NVSHMEM_ROOT}"
        PATH_SUFFIXES lib lib64)

    if(NVSHMEM_INCLUDE_DIR AND
       EXISTS "${NVSHMEM_INCLUDE_DIR}/nvshmemx.h" AND
       NVSHMEM_HOST_LIBRARY AND
       NVSHMEM_DEVICE_LIBRARY)
        set(NVSHMEM_HOST_TARGET "${NVSHMEM_HOST_LIBRARY}")
        set(NVSHMEM_DEVICE_TARGET "${NVSHMEM_DEVICE_LIBRARY}")
    endif()
endif()

find_package_handle_standard_args(NVSHMEM
    REQUIRED_VARS NVSHMEM_HOST_TARGET NVSHMEM_DEVICE_TARGET)

if(NVSHMEM_FOUND)
    if(NOT TARGET NVSHMEM::host)
        add_library(NVSHMEM::host INTERFACE IMPORTED GLOBAL)
        set_target_properties(NVSHMEM::host PROPERTIES
            INTERFACE_LINK_LIBRARIES "${NVSHMEM_HOST_TARGET}")
        if(NVSHMEM_INCLUDE_DIR)
            set_target_properties(NVSHMEM::host PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIR}")
        endif()
    endif()

    if(NOT TARGET NVSHMEM::device)
        add_library(NVSHMEM::device INTERFACE IMPORTED GLOBAL)
        set_target_properties(NVSHMEM::device PROPERTIES
            INTERFACE_LINK_LIBRARIES "${NVSHMEM_DEVICE_TARGET}")
        if(NVSHMEM_INCLUDE_DIR)
            set_target_properties(NVSHMEM::device PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIR}")
        endif()
    endif()
endif()

mark_as_advanced(
    NVSHMEM_INCLUDE_DIR
    NVSHMEM_HOST_LIBRARY
    NVSHMEM_DEVICE_LIBRARY)
