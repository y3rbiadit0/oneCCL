#
# Copyright 2016-2020 Intel Corporation
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

set(test_name nccl_grouped_pt2pt_test)

add_executable(${test_name} ${test_name}.cpp)
target_compile_features(${test_name} PRIVATE cxx_std_17)
target_include_directories(
    ${test_name}
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${MPI_INCLUDE_DIR}
)
target_link_directories(${test_name} PRIVATE ${MPI_LIB_DIR})
target_link_libraries(${test_name} PRIVATE ccl mpi ${COMPUTE_BACKEND_TARGET_NAME} rt m)

find_program(
    MPIEXEC_EXECUTABLE
    NAMES mpiexec
    HINTS "${MPI_DIR}/bin" "${PROJECT_SOURCE_DIR}/deps/mpi/bin"
    REQUIRED
)

add_test(
    NAME ${test_name}
    COMMAND ${MPIEXEC_EXECUTABLE} -n 2 $<TARGET_FILE:${test_name}>
)
set_tests_properties(
    ${test_name}
    PROPERTIES
        ENVIRONMENT "CCL_BACKEND=nccl"
        LABELS "nccl;gpu"
        PROCESSORS 2
        RUN_SERIAL TRUE
)
