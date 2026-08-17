#
# Copyright 2026 Contributors
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

set(test_name oshmpi_collectives_test)

find_package(MPI REQUIRED COMPONENTS C)

add_executable(${test_name} ${test_name}.cpp)
target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${test_name} PRIVATE ccl MPI::MPI_C OSHMPI::oshmpi rt m)

# The project builds with CMAKE_SKIP_RPATH, so the test binary records no path to
# libccl.so.1 or liboshmpi.so. Hand the loader both directories for ctest runs.
get_filename_component(_oshmpi_test_lib_dir "${OSHMPI_LIBRARY}" DIRECTORY)
set(_oshmpi_test_ld_path "${CCL_BUILD_DIR}:${_oshmpi_test_lib_dir}")
# An empty trailing entry would make the loader search the current directory.
if (NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
    set(_oshmpi_test_ld_path "${_oshmpi_test_ld_path}:$ENV{LD_LIBRARY_PATH}")
endif()

# Device coverage, driven entirely through SYCL - the way oneCCL is used on
# device. Covers the device communicator, stream draining, staging of device
# operands, group scope and send/recv, none of which the host test reaches.
# Needs the SYCL flags on the target itself, since it builds kernels of its own.
if (CCL_ENABLE_SYCL)
    set(sycl_test_name oshmpi_sycl_collectives_test)
    add_executable(${sycl_test_name} ${sycl_test_name}.cpp)
    target_compile_features(${sycl_test_name} PRIVATE cxx_std_17)
    target_include_directories(${sycl_test_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${sycl_test_name} PRIVATE ccl MPI::MPI_C OSHMPI::oshmpi rt m)

    add_test(
        NAME ${sycl_test_name}
        COMMAND ${MPIEXEC_EXECUTABLE} -n 2 $<TARGET_FILE:${sycl_test_name}>
    )
    set_tests_properties(
        ${sycl_test_name}
        PROPERTIES
            ENVIRONMENT "CCL_BACKEND=oshmpi;CCL_OSHMPI_STAGING_SIZE=128;SHMEM_SYMMETRIC_SIZE=128M;LD_LIBRARY_PATH=${_oshmpi_test_ld_path}"
            LABELS "oshmpi;sycl"
            PROCESSORS 2
            RUN_SERIAL TRUE
            TIMEOUT 180
    )
endif()

add_test(
    NAME ${test_name}
    COMMAND ${MPIEXEC_EXECUTABLE} -n 2 $<TARGET_FILE:${test_name}>
)
set_tests_properties(
    ${test_name}
    PROPERTIES
        ENVIRONMENT "CCL_BACKEND=oshmpi;CCL_OSHMPI_STAGING_SIZE=128;SHMEM_SYMMETRIC_SIZE=128M;LD_LIBRARY_PATH=${_oshmpi_test_ld_path}"
        LABELS "oshmpi;host"
        PROCESSORS 2
        RUN_SERIAL TRUE
        TIMEOUT 120
)
