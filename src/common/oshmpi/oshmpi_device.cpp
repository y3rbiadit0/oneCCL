/*
 Copyright 2026 Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
#ifdef CCL_ENABLE_OSHMPI

#include "common/oshmpi/oshmpi_device.hpp"

#include "oneapi/ccl.hpp"
#include "common/log/log.hpp"

#ifdef CCL_ENABLE_OSHMPI_CUDA
#include <cuda_runtime.h>
#endif // CCL_ENABLE_OSHMPI_CUDA

namespace ccl {
namespace oshmpi_device {

#ifdef CCL_ENABLE_OSHMPI_CUDA

namespace {

void check(cudaError_t status, const char* operation) {
    CCL_THROW_IF_NOT(status == cudaSuccess,
                     operation,
                     " failed: ",
                     cudaGetErrorString(status));
}

} // namespace

bool enabled() noexcept {
    return true;
}

location classify(const void* buffer) noexcept {
    if (!buffer) {
        return location::host;
    }

    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, buffer);
    if (status != cudaSuccess) {
        // An unregistered host allocation reports an error on older CUDA
        // versions. Clear the sticky error so the next real CUDA call is not
        // blamed for it, and treat the buffer as host memory.
        cudaGetLastError();
        return location::unknown;
    }

    switch (attributes.type) {
        case cudaMemoryTypeDevice:
        case cudaMemoryTypeManaged: return location::device;
        case cudaMemoryTypeHost:
        case cudaMemoryTypeUnregistered: return location::host;
        default: return location::unknown;
    }
}

void copy_device_to_host(void* destination, const void* source, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    check(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
}

void copy_host_to_device(void* destination, const void* source, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    check(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
}

void copy_device_to_device(void* destination, const void* source, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    check(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy D2D");
}

void synchronize() {
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

bool try_register_host_memory(void* buffer, std::size_t bytes) noexcept {
    if (!buffer || bytes == 0) {
        return false;
    }
    const cudaError_t status = cudaHostRegister(buffer, bytes, cudaHostRegisterDefault);
    if (status != cudaSuccess) {
        // Clear the sticky error so the next real CUDA call is not blamed for it.
        cudaGetLastError();
        LOG_INFO("could not pin the OSHMPI staging arena (",
                 cudaGetErrorString(status),
                 "); staged copies will use the pageable path");
        return false;
    }
    return true;
}

void unregister_host_memory(void* buffer) noexcept {
    if (!buffer) {
        return;
    }
    if (cudaHostUnregister(buffer) != cudaSuccess) {
        cudaGetLastError();
    }
}

#else // CCL_ENABLE_OSHMPI_CUDA

bool enabled() noexcept {
    return false;
}

location classify(const void*) noexcept {
    return location::host;
}

namespace {

[[noreturn]] void unavailable(const char* operation) {
    CCL_THROW(operation,
              " requires a CUDA-enabled OSHMPI backend; rebuild with CCL_ENABLE_OSHMPI_CUDA=ON");
}

} // namespace

void copy_device_to_host(void*, const void*, std::size_t) {
    unavailable("device to host copy");
}

void copy_host_to_device(void*, const void*, std::size_t) {
    unavailable("host to device copy");
}

void copy_device_to_device(void*, const void*, std::size_t) {
    unavailable("device to device copy");
}

void synchronize() {
    unavailable("device synchronize");
}

/* Not an error without CUDA: there are no device buffers to copy to or from, so
 * there is nothing for pinning to accelerate. */
bool try_register_host_memory(void*, std::size_t) noexcept {
    return false;
}

void unregister_host_memory(void*) noexcept {}

#endif // CCL_ENABLE_OSHMPI_CUDA

} // namespace oshmpi_device
} // namespace ccl

#endif // CCL_ENABLE_OSHMPI
