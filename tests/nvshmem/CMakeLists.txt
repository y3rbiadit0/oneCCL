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

set(test_name nvshmem_m1_comm_test)

add_executable(${test_name} ${test_name}.cpp)
target_compile_features(${test_name} PRIVATE cxx_std_17)
target_include_directories(
    ${test_name}
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${MPI_INCLUDE_DIR}
)
target_link_directories(${test_name} PRIVATE ${MPI_LIB_DIR})
target_link_libraries(${test_name} PRIVATE ccl mpi ${COMPUTE_BACKEND_TARGET_NAME} rt m)

if (MPI_DIR)
    set(nvshmem_m1_mpi_bin_dir "${MPI_DIR}/bin")
else()
    set(nvshmem_m1_mpi_bin_dir "${PROJECT_SOURCE_DIR}/deps/mpi/bin")
endif()

find_program(
    NVSHMEM_M1_MPIEXEC_EXECUTABLE
    NAMES mpiexec mpiexec.hydra mpirun
    HINTS "${nvshmem_m1_mpi_bin_dir}"
    NO_DEFAULT_PATH
    REQUIRED
)

add_test(
    NAME ${test_name}
    COMMAND ${CMAKE_COMMAND} -E env --unset=NVSHMEM_BOOTSTRAP
            ${NVSHMEM_M1_MPIEXEC_EXECUTABLE} -n 2 $<TARGET_FILE:${test_name}>
)
set(nvshmem_m1_test_environment
    "CCL_BACKEND=nvshmem"
    "CCL_ATL_TRANSPORT=mpi"
    "CCL_MPI_LIBRARY_PATH=${MPI_LIB_DIR}/libmpi.so.12"
    "I_MPI_HYDRA_BOOTSTRAP=slurm"
    "I_MPI_FABRICS=shm"
    "NVSHMEM_REMOTE_TRANSPORT=ibrc"
    "NVSHMEM_IB_ENABLE_IBGDA=0"
    "NVSHMEM_DISABLE_NCCL=1"
    "NVSHMEM_IB_SL=1"
    "NVSHMEM_SYMMETRIC_SIZE=1G"
    "CCL_NVSHMEM_VALIDATE_STAGING=1"
    "LD_LIBRARY_PATH=${CCL_BUILD_DIR}:${MPI_LIB_DIR}:$ENV{LD_LIBRARY_PATH}"
)
set_tests_properties(
    ${test_name}
    PROPERTIES
        ENVIRONMENT
            "${nvshmem_m1_test_environment};CCL_NVSHMEM_STAGING_SIZE=1M"
        LABELS "nvshmem;cuda;gpu"
        PROCESSORS 2
        RUN_SERIAL TRUE
        TIMEOUT 180
)

set(invalid_staging_test_name nvshmem_m2_invalid_staging_test)
add_test(
    NAME ${invalid_staging_test_name}
    COMMAND ${CMAKE_COMMAND}
            -DMPIEXEC=${NVSHMEM_M1_MPIEXEC_EXECUTABLE}
            -DTEST_EXECUTABLE=$<TARGET_FILE:${test_name}>
            "-DEXPECTED_TEXT=CCL_NVSHMEM_STAGING_SIZE has no numeric value"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake
)
set_tests_properties(
    ${invalid_staging_test_name}
    PROPERTIES
        ENVIRONMENT
            "${nvshmem_m1_test_environment};CCL_NVSHMEM_STAGING_SIZE=invalid"
        LABELS "nvshmem;cuda;gpu"
        PROCESSORS 2
        RUN_SERIAL TRUE
        TIMEOUT 180
)
