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

#include "nvshmem_uid_adapter.h"

#include <cuda_runtime_api.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstdio>
#include <cstring>

namespace {
thread_local char last_error[512] = {};
bool initialized = false;

nvshmemx_uniqueid_t make_unique_id() {
#ifdef NVSHMEMX_UNIQUEID_INITIALIZER
    return NVSHMEMX_UNIQUEID_INITIALIZER;
#else
    return {};
#endif
}

nvshmemx_init_attr_t make_init_attributes() {
#ifdef NVSHMEMX_INIT_ATTR_INITIALIZER
    return NVSHMEMX_INIT_ATTR_INITIALIZER;
#else
    return {};
#endif
}

void clear_error() {
    last_error[0] = '\0';
}

int set_cuda_error(const char* operation, cudaError_t status) {
    std::snprintf(last_error,
                  sizeof(last_error),
                  "%s failed: %s",
                  operation,
                  cudaGetErrorString(status));
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
} // namespace

extern "C" size_t oneccl_nvshmem_uid_size(void) {
    return sizeof(nvshmemx_uniqueid_t);
}

extern "C" int oneccl_nvshmem_uid_get(void* uid, size_t size) {
    clear_error();
    if (uid == nullptr || size != sizeof(nvshmemx_uniqueid_t)) {
        std::snprintf(last_error, sizeof(last_error), "invalid NVSHMEM UID buffer");
        return -1;
    }

    nvshmemx_uniqueid_t native_uid = make_unique_id();
    const int status = nvshmemx_get_uniqueid(&native_uid);
    if (status != 0) {
        return set_nvshmem_error("nvshmemx_get_uniqueid", status);
    }
    std::memcpy(uid, &native_uid, sizeof(native_uid));
    return 0;
}

extern "C" int oneccl_nvshmem_uid_cuda_device_count(int* count) {
    clear_error();
    if (count == nullptr) {
        std::snprintf(last_error, sizeof(last_error), "CUDA device count output is null");
        return -1;
    }
    const cudaError_t status = cudaGetDeviceCount(count);
    return (status == cudaSuccess) ? 0 : set_cuda_error("cudaGetDeviceCount", status);
}

extern "C" int oneccl_nvshmem_uid_cuda_set_device(int device) {
    clear_error();
    const cudaError_t status = cudaSetDevice(device);
    return (status == cudaSuccess) ? 0 : set_cuda_error("cudaSetDevice", status);
}

extern "C" int oneccl_nvshmem_uid_init(int rank,
                                         int size,
                                         const void* uid,
                                         size_t uid_size) {
    clear_error();
    if (initialized) {
        std::snprintf(last_error, sizeof(last_error), "NVSHMEM host library is already initialized");
        return -1;
    }
    if (rank < 0 || size <= 0 || rank >= size) {
        std::snprintf(last_error, sizeof(last_error), "invalid NVSHMEM PE topology");
        return -1;
    }
    if (uid == nullptr || uid_size != sizeof(nvshmemx_uniqueid_t)) {
        std::snprintf(last_error, sizeof(last_error), "invalid NVSHMEM UID buffer");
        return -1;
    }

    nvshmemx_uniqueid_t native_uid = make_unique_id();
    std::memcpy(&native_uid, uid, sizeof(native_uid));
    nvshmemx_init_attr_t attributes = make_init_attributes();
    nvshmemx_set_attr_uniqueid_args(rank, size, &native_uid, &attributes);
    const int status =
        nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attributes);
    if (status != 0) {
        return set_nvshmem_error("nvshmemx_hostlib_init_attr", status);
    }
    initialized = true;
    return 0;
}

extern "C" int oneccl_nvshmem_uid_is_initialized(void) {
    return nvshmemx_init_status() >= NVSHMEM_STATUS_IS_INITIALIZED;
}

extern "C" int oneccl_nvshmem_uid_my_pe(void) {
    return nvshmem_my_pe();
}

extern "C" int oneccl_nvshmem_uid_n_pes(void) {
    return nvshmem_n_pes();
}

extern "C" void* oneccl_nvshmem_uid_malloc(size_t size) {
    clear_error();
    void* ptr = nvshmem_malloc(size);
    if (ptr == nullptr && size != 0) {
        std::snprintf(last_error, sizeof(last_error), "nvshmem_malloc returned null");
    }
    return ptr;
}

extern "C" void oneccl_nvshmem_uid_barrier_all(void) {
    nvshmem_barrier_all();
}

extern "C" void oneccl_nvshmem_uid_free(void* ptr) {
    if (ptr != nullptr) {
        nvshmem_free(ptr);
    }
}

extern "C" void oneccl_nvshmem_uid_finalize(void) {
    if (initialized) {
        nvshmemx_hostlib_finalize();
        initialized = false;
    }
}

extern "C" const char* oneccl_nvshmem_uid_last_error(void) {
    return last_error;
}
