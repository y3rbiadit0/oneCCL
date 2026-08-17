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
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Not Intel MPI: OSHMPI carries whatever MPI it was built against, and these have
# to resolve the same one.
find_package(MPI REQUIRED COMPONENTS C)

set(oshmpi_examples
    oshmpi_allreduce_test.cpp
    oshmpi_allgather_test.cpp
    oshmpi_broadcast_test.cpp
)

if (CCL_ENABLE_SYCL)
    list(APPEND oshmpi_examples oshmpi_sycl_allreduce_test.cpp)
endif()

# Deliberately not EXAMPLES_INC_DIRS / EXAMPLES_LIB_DIRS: those carry the vendored
# Intel MPI, which would be compiled against while MPI::MPI_C links another MPI.
foreach(src ${oshmpi_examples})
    get_filename_component(executable ${src} NAME_WE)
    add_executable(${executable} ${src})
    target_include_directories(${executable}
        PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/examples/include)
    target_link_libraries(${executable} PRIVATE ccl MPI::MPI_C rt m)
    if (COMPUTE_BACKEND_TARGET_NAME)
        target_link_libraries(${executable} PRIVATE ${COMPUTE_BACKEND_TARGET_NAME})
    endif()
    install(TARGETS ${executable}
            RUNTIME DESTINATION ${CCL_INSTALL_EXAMPLES}/oshmpi OPTIONAL)
endforeach()
