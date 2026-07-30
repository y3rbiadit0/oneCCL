/*
 * Copyright 2026 UXL Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nvshmem_m0_adapter.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstring>
#include <cstdio>
#include <vector>

namespace {
thread_local char last_error[512] = {};
bool mpi_initialized = false;
bool mpi_owned = false;
bool nvshmem_initialized = false;

int set_cuda_error(const char* operation, cudaError_t status) {
    std::snprintf(last_error,
                  sizeof(last_error),
                  "%s failed: %s",
                  operation,
                  cudaGetErrorString(status));
    return static_cast<int>(status);
}

int set_cuda_driver_error(const char* operation, CUresult status) {
    const char* message = nullptr;
    cuGetErrorString(status, &message);
    std::snprintf(last_error,
                  sizeof(last_error),
                  "%s failed: %s (CUDA driver status %d)",
                  operation,
                  (message != nullptr) ? message : "unknown error",
                  static_cast<int>(status));
    return static_cast<int>(status);
}

int set_nvshmem_error(const char* operation, int status) {
    std::snprintf(last_error,
                  sizeof(last_error),
                  "%s failed with NVSHMEM status %d",
                  operation,
                  status);
    return status;
}

int set_mpi_error(const char* operation, int status) {
    char message[MPI_MAX_ERROR_STRING] = {};
    int message_length = 0;
    MPI_Error_string(status, message, &message_length);
    std::snprintf(last_error,
                  sizeof(last_error),
                  "%s failed: %.*s (MPI status %d)",
                  operation,
                  message_length,
                  message,
                  status);
    return status;
}

void clear_error() {
    last_error[0] = '\0';
}
} // namespace

extern "C" int oneccl_nvshmem_m0_mpi_init(int* argc,
                                            char*** argv,
                                            int* rank,
                                            int* size,
                                            int* node_rank) {
    clear_error();
    if (rank == nullptr || size == nullptr || node_rank == nullptr) {
        std::snprintf(last_error, sizeof(last_error), "MPI topology output is null");
        return -1;
    }

    int finalized = 0;
    int status = MPI_Finalized(&finalized);
    if (status != MPI_SUCCESS) {
        return set_mpi_error("MPI_Finalized", status);
    }
    if (finalized != 0) {
        std::snprintf(last_error, sizeof(last_error), "MPI was already finalized");
        return -1;
    }

    int initialized = 0;
    status = MPI_Initialized(&initialized);
    if (status != MPI_SUCCESS) {
        return set_mpi_error("MPI_Initialized", status);
    }
    if (initialized == 0) {
        status = MPI_Init(argc, argv);
        if (status != MPI_SUCCESS) {
            return set_mpi_error("MPI_Init", status);
        }
        mpi_owned = true;
    }
    mpi_initialized = true;

    status = MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    if (status != MPI_SUCCESS) {
        return set_mpi_error("MPI_Comm_set_errhandler", status);
    }
    status = MPI_Comm_rank(MPI_COMM_WORLD, rank);
    if (status != MPI_SUCCESS) {
        return set_mpi_error("MPI_Comm_rank", status);
    }
    status = MPI_Comm_size(MPI_COMM_WORLD, size);
    if (status != MPI_SUCCESS) {
        return set_mpi_error("MPI_Comm_size", status);
    }

    MPI_Comm node_comm = MPI_COMM_NULL;
    status = MPI_Comm_split_type(
        MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, *rank, MPI_INFO_NULL, &node_comm);
    if (status != MPI_SUCCESS) {
        return set_mpi_error("MPI_Comm_split_type", status);
    }
    status = MPI_Comm_rank(node_comm, node_rank);
    const int free_status = MPI_Comm_free(&node_comm);
    if (status != MPI_SUCCESS) {
        return set_mpi_error("MPI node-local rank", status);
    }
    if (free_status != MPI_SUCCESS) {
        return set_mpi_error("MPI_Comm_free", free_status);
    }
    return 0;
}

extern "C" int oneccl_nvshmem_m0_init(void) {
    clear_error();
    if (!mpi_initialized) {
        std::snprintf(last_error, sizeof(last_error), "MPI must be initialized first");
        return -1;
    }

    nvshmemx_init_attr_t attributes = {};
    MPI_Comm world = MPI_COMM_WORLD;
    attributes.mpi_comm = &world;
    const int status =
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attributes);
    if (status != 0) {
        return set_nvshmem_error("nvshmemx_init_attr", status);
    }
    nvshmem_initialized = true;
    return status;
}

extern "C" int oneccl_nvshmem_m0_my_pe(void) {
    return nvshmem_my_pe();
}

extern "C" int oneccl_nvshmem_m0_n_pes(void) {
    return nvshmem_n_pes();
}

extern "C" int oneccl_nvshmem_m0_node_pe(void) {
    return nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
}

extern "C" int oneccl_nvshmem_m0_cuda_device_count(int* count) {
    clear_error();
    if (count == nullptr) {
        std::snprintf(last_error, sizeof(last_error), "CUDA device count output is null");
        return -1;
    }

    const cudaError_t status = cudaGetDeviceCount(count);
    return (status == cudaSuccess) ? 0 : set_cuda_error("cudaGetDeviceCount", status);
}

extern "C" int oneccl_nvshmem_m0_cuda_set_device(int device) {
    clear_error();
    const cudaError_t status = cudaSetDevice(device);
    return (status == cudaSuccess) ? 0 : set_cuda_error("cudaSetDevice", status);
}

extern "C" int oneccl_nvshmem_m0_validate_cuda_device(int device) {
    clear_error();
    CUdevice native_device = 0;
    CUresult driver_status = cuDeviceGet(&native_device, device);
    if (driver_status != CUDA_SUCCESS) {
        return set_cuda_driver_error("cuDeviceGet", driver_status);
    }
    CUuuid local_uuid = {};
    driver_status = cuDeviceGetUuid(&local_uuid, native_device);
    if (driver_status != CUDA_SUCCESS) {
        return set_cuda_driver_error("cuDeviceGetUuid", driver_status);
    }

    MPI_Comm node_comm = MPI_COMM_NULL;
    int mpi_status =
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    if (mpi_status != MPI_SUCCESS) {
        return set_mpi_error("MPI_Comm_split_type for CUDA validation", mpi_status);
    }

    int node_size = 0;
    mpi_status = MPI_Comm_size(node_comm, &node_size);
    if (mpi_status != MPI_SUCCESS) {
        MPI_Comm_free(&node_comm);
        return set_mpi_error("MPI node-local size", mpi_status);
    }

    std::vector<CUuuid> uuids(static_cast<size_t>(node_size));
    constexpr int uuid_size = static_cast<int>(sizeof(CUuuid));
    mpi_status = MPI_Allgather(&local_uuid,
                               uuid_size,
                               MPI_BYTE,
                               uuids.data(),
                               uuid_size,
                               MPI_BYTE,
                               node_comm);
    const int free_status = MPI_Comm_free(&node_comm);
    if (mpi_status != MPI_SUCCESS) {
        return set_mpi_error("MPI_Allgather for CUDA UUIDs", mpi_status);
    }
    if (free_status != MPI_SUCCESS) {
        return set_mpi_error("MPI_Comm_free for CUDA validation", free_status);
    }

    for (int left = 0; left < node_size; left++) {
        for (int right = left + 1; right < node_size; right++) {
            if (std::memcmp(uuids[left].bytes,
                            uuids[right].bytes,
                            sizeof(uuids[left].bytes)) == 0) {
                std::snprintf(last_error,
                              sizeof(last_error),
                              "multiple MPI ranks selected the same physical CUDA device");
                return -1;
            }
        }
    }
    return 0;
}

extern "C" void* oneccl_nvshmem_m0_malloc(size_t size) {
    clear_error();
    void* ptr = nvshmem_malloc(size);
    if (ptr == nullptr && size != 0) {
        std::snprintf(last_error,
                      sizeof(last_error),
                      "nvshmem_malloc failed to allocate %zu bytes",
                      size);
    }
    return ptr;
}

extern "C" void oneccl_nvshmem_m0_free(void* ptr) {
    clear_error();
    nvshmem_free(ptr);
}

extern "C" int oneccl_nvshmem_m0_copy_async(void* destination,
                                              const void* source,
                                              size_t size,
                                              void* native_stream) {
    clear_error();
    if (destination == nullptr || source == nullptr) {
        std::snprintf(last_error, sizeof(last_error), "CUDA copy received a null buffer");
        return -1;
    }

    const cudaError_t status = cudaMemcpyAsync(destination,
                                               source,
                                               size,
                                               cudaMemcpyDefault,
                                               reinterpret_cast<cudaStream_t>(native_stream));
    return (status == cudaSuccess) ? 0 : set_cuda_error("cudaMemcpyAsync", status);
}

extern "C" int oneccl_nvshmem_m0_stream_synchronize(void* native_stream) {
    clear_error();
    const cudaError_t status =
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(native_stream));
    return (status == cudaSuccess) ? 0 : set_cuda_error("cudaStreamSynchronize", status);
}

extern "C" int oneccl_nvshmem_m0_float_sum_reduce(float* destination,
                                                    const float* source,
                                                    size_t count,
                                                    void* native_stream) {
    clear_error();
    if (destination == nullptr || source == nullptr) {
        std::snprintf(last_error,
                      sizeof(last_error),
                      "NVSHMEM allreduce received a null buffer");
        return -1;
    }

    const int status = nvshmemx_float_sum_reduce_on_stream(
        NVSHMEM_TEAM_WORLD,
        destination,
        source,
        count,
        reinterpret_cast<cudaStream_t>(native_stream));
    return (status == 0) ? 0 : set_nvshmem_error("nvshmemx_float_sum_reduce_on_stream", status);
}

extern "C" int oneccl_nvshmem_m0_double_min_reduce(double* destination,
                                                     const double* source,
                                                     size_t count,
                                                     void* native_stream) {
    clear_error();
    if (destination == nullptr || source == nullptr) {
        std::snprintf(last_error,
                      sizeof(last_error),
                      "NVSHMEM minimum reduction received a null buffer");
        return -1;
    }

    const int status = nvshmemx_double_min_reduce_on_stream(
        NVSHMEM_TEAM_WORLD,
        destination,
        source,
        count,
        reinterpret_cast<cudaStream_t>(native_stream));
    return (status == 0) ? 0 : set_nvshmem_error("nvshmemx_double_min_reduce_on_stream", status);
}

extern "C" int oneccl_nvshmem_m0_double_max_reduce(double* destination,
                                                     const double* source,
                                                     size_t count,
                                                     void* native_stream) {
    clear_error();
    if (destination == nullptr || source == nullptr) {
        std::snprintf(last_error,
                      sizeof(last_error),
                      "NVSHMEM maximum reduction received a null buffer");
        return -1;
    }

    const int status = nvshmemx_double_max_reduce_on_stream(
        NVSHMEM_TEAM_WORLD,
        destination,
        source,
        count,
        reinterpret_cast<cudaStream_t>(native_stream));
    return (status == 0) ? 0 : set_nvshmem_error("nvshmemx_double_max_reduce_on_stream", status);
}

extern "C" int oneccl_nvshmem_m0_barrier(void* native_stream) {
    clear_error();
    const int status = nvshmemx_barrier_on_stream(
        NVSHMEM_TEAM_WORLD,
        reinterpret_cast<cudaStream_t>(native_stream));
    return (status == 0) ? 0 : set_nvshmem_error("nvshmemx_barrier_on_stream", status);
}

extern "C" void oneccl_nvshmem_m0_finalize(void) {
    clear_error();
    if (nvshmem_initialized) {
        nvshmem_finalize();
        nvshmem_initialized = false;
    }
    if (mpi_initialized && mpi_owned) {
        MPI_Finalize();
    }
    mpi_initialized = false;
    mpi_owned = false;
}

extern "C" void oneccl_nvshmem_m0_global_exit(int status) {
    if (nvshmem_initialized) {
        nvshmem_global_exit(status);
    }
    if (mpi_initialized) {
        MPI_Abort(MPI_COMM_WORLD, status);
    }
}

extern "C" const char* oneccl_nvshmem_m0_last_error(void) {
    return last_error;
}
