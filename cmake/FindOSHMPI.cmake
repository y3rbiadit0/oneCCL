# Find the OSHMPI library and includes
#
# OSHMPI_INCLUDE_DIR - where to find shmem.h
# OSHMPI_LIBRARY - OSHMPI shared library
# OSHMPI_CUDART_LIBRARY - CUDA runtime required by CUDA-enabled OSHMPI builds
# OSHMPI_FOUND - true if OSHMPI was found and validated
#
# OSHMPI is OpenSHMEM over MPI, so it carries its own MPI dependency. That MPI is
# discovered here with the standard FindMPI module and is deliberately independent
# of oneCCL's MPI_DIR/MPI_INCLUDE_DIR, which describe the vendored Intel MPI used
# by the native ATL transport. Point this at a specific MPI with MPI_C_COMPILER.

find_package(MPI REQUIRED COMPONENTS C)

set(_OSHMPI_ROOT_HINTS
    "${OSHMPI_ROOT}"
    "$ENV{OSHMPI_ROOT}"
    "${OSHMPI_HOME}"
    "$ENV{OSHMPI_HOME}"
    "${OSHMPI_PREFIX}"
    "$ENV{OSHMPI_PREFIX}")

find_path(OSHMPI_INCLUDE_DIR
    NAMES shmem.h
    HINTS ${_OSHMPI_ROOT_HINTS}
    PATH_SUFFIXES include)

set(_OSHMPI_ORIGINAL_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX})
find_library(OSHMPI_LIBRARY
    NAMES oshmpi
    HINTS ${_OSHMPI_ROOT_HINTS}
    PATH_SUFFIXES lib lib64)
set(CMAKE_FIND_LIBRARY_SUFFIXES ${_OSHMPI_ORIGINAL_LIBRARY_SUFFIXES})

find_library(OSHMPI_CUDART_LIBRARY
    NAMES cudart
    HINTS
        "${CUDA_ROOT}"
        "$ENV{CUDA_ROOT}"
        "${CUDA_HOME}"
        "$ENV{CUDA_HOME}"
    PATH_SUFFIXES lib64 lib)

set(_OSHMPI_LINK_LIBRARIES ${OSHMPI_LIBRARY} MPI::MPI_C)
if (OSHMPI_CUDART_LIBRARY)
    list(APPEND _OSHMPI_LINK_LIBRARIES ${OSHMPI_CUDART_LIBRARY})
endif()

# Re-run the probe when the OSHMPI install or its MPI changes, so switching MPI
# modules cannot silently reuse a stale cached result.
set(_OSHMPI_CHECK_FINGERPRINT
    "${OSHMPI_INCLUDE_DIR}|${OSHMPI_LIBRARY}|${OSHMPI_CUDART_LIBRARY}|${MPI_C_COMPILER}")
if (NOT "${OSHMPI_CHECK_FINGERPRINT}" STREQUAL "${_OSHMPI_CHECK_FINGERPRINT}")
    unset(OSHMPI_COMPILES_AND_LINKS CACHE)
endif()
set(OSHMPI_CHECK_FINGERPRINT "${_OSHMPI_CHECK_FINGERPRINT}" CACHE INTERNAL
    "OSHMPI compile/link check inputs" FORCE)

if (OSHMPI_INCLUDE_DIR AND OSHMPI_LIBRARY)
    include(CheckCXXSourceCompiles)
    set(_OSHMPI_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
    set(_OSHMPI_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
    set(CMAKE_REQUIRED_INCLUDES ${OSHMPI_INCLUDE_DIR} ${MPI_C_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES ${_OSHMPI_LINK_LIBRARIES})
    check_cxx_source_compiles([=[
#include <stdint.h>
#include <shmem.h>

#ifndef OSHMPI_PRESERVE_EXTERNAL_MPI
#error "OSHMPI must include the external MPI ownership patch"
#endif

int main(void) {
    int provided = SHMEM_THREAD_SINGLE;
    int8_t i8 = 0;
    uint8_t u8 = 0;
    int16_t i16 = 0;
    uint16_t u16 = 0;
    int32_t i32 = 0;
    uint32_t u32 = 0;
    int64_t i64 = 0;
    uint64_t u64 = 0;
    float f32 = 0;
    double f64 = 0;

    if (oshmpi_preserves_external_mpi() != 1) return 1;
    shmem_init_thread(SHMEM_THREAD_SERIALIZED, &provided);
    shmem_barrier_all();
    shmem_fcollectmem(SHMEM_TEAM_WORLD, &i32, &i32, 0);
    shmem_alltoallmem(SHMEM_TEAM_WORLD, &i32, &i32, 0);
    shmem_broadcastmem(SHMEM_TEAM_WORLD, &i32, &i32, 0, 0);
    shmem_int8_sum_reduce(SHMEM_TEAM_WORLD, &i8, &i8, 0);
    shmem_uint8_prod_reduce(SHMEM_TEAM_WORLD, &u8, &u8, 0);
    shmem_int16_min_reduce(SHMEM_TEAM_WORLD, &i16, &i16, 0);
    shmem_uint16_max_reduce(SHMEM_TEAM_WORLD, &u16, &u16, 0);
    shmem_int32_sum_reduce(SHMEM_TEAM_WORLD, &i32, &i32, 0);
    shmem_uint32_prod_reduce(SHMEM_TEAM_WORLD, &u32, &u32, 0);
    shmem_int64_min_reduce(SHMEM_TEAM_WORLD, &i64, &i64, 0);
    shmem_uint64_max_reduce(SHMEM_TEAM_WORLD, &u64, &u64, 0);
    shmem_float_sum_reduce(SHMEM_TEAM_WORLD, &f32, &f32, 0);
    shmem_double_max_reduce(SHMEM_TEAM_WORLD, &f64, &f64, 0);
    void* ptr = shmem_malloc(64);
    shmem_free(ptr);
    shmem_finalize();
    return 0;
}
]=] OSHMPI_COMPILES_AND_LINKS)
    set(CMAKE_REQUIRED_INCLUDES ${_OSHMPI_REQUIRED_INCLUDES})
    set(CMAKE_REQUIRED_LIBRARIES ${_OSHMPI_REQUIRED_LIBRARIES})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OSHMPI
    REQUIRED_VARS OSHMPI_INCLUDE_DIR OSHMPI_LIBRARY OSHMPI_COMPILES_AND_LINKS)

if (OSHMPI_FOUND)
    set(OSHMPI_INCLUDE_DIRS ${OSHMPI_INCLUDE_DIR})
    set(OSHMPI_LIBRARIES ${_OSHMPI_LINK_LIBRARIES})

    if (NOT TARGET OSHMPI::oshmpi)
        # MPI::MPI_C carries OSHMPI's MPI include dirs and link flags transitively,
        # so consumers get the same MPI that OSHMPI was validated against.
        set(_OSHMPI_INTERFACE_LIBRARIES MPI::MPI_C)
        if (OSHMPI_CUDART_LIBRARY)
            list(APPEND _OSHMPI_INTERFACE_LIBRARIES "${OSHMPI_CUDART_LIBRARY}")
        endif()

        add_library(OSHMPI::oshmpi SHARED IMPORTED)
        set_target_properties(OSHMPI::oshmpi PROPERTIES
            IMPORTED_LOCATION "${OSHMPI_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${OSHMPI_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${_OSHMPI_INTERFACE_LIBRARIES}")
    endif()
endif()

mark_as_advanced(
    OSHMPI_INCLUDE_DIR
    OSHMPI_LIBRARY
    OSHMPI_CUDART_LIBRARY
    OSHMPI_COMPILES_AND_LINKS)
