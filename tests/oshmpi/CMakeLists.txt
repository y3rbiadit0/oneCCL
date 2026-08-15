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

add_test(
    NAME ${test_name}
    COMMAND ${MPIEXEC_EXECUTABLE} -n 2 $<TARGET_FILE:${test_name}>
)
set_tests_properties(
    ${test_name}
    PROPERTIES
        ENVIRONMENT "CCL_BACKEND=oshmpi;CCL_OSHMPI_STAGING_SIZE=128;SHMEM_SYMMETRIC_SIZE=128M"
        LABELS "oshmpi;host"
        PROCESSORS 2
        RUN_SERIAL TRUE
        TIMEOUT 120
)
