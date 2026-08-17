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

#ifdef CCL_ENABLE_OSHMPI_PINNED_STAGING
#include <cuda_runtime.h>
#endif // CCL_ENABLE_OSHMPI_PINNED_STAGING

namespace ccl {
namespace oshmpi_device {

#ifdef CCL_ENABLE_SYCL

bool accessor::enabled() const noexcept {
    return sycl_context.has_value();
}

location accessor::classify(const void* buffer) const noexcept {
    if (!buffer || !sycl_context) {
        return location::host;
    }

    /* get_pointer_type reports `unknown` for allocations it does not own, but a
     * foreign context can still make the runtime throw. Any failure means "not
     * device memory as far as this context is concerned", which is the safe
     * answer: it routes the buffer down the host path. */
    try {
        switch (sycl::get_pointer_type(buffer, *sycl_context)) {
            /* Shared USM is reachable from the host, so a plain memcpy would also
             * be correct - but it is device-resident often enough that letting the
             * runtime move it is the better default. */
            case sycl::usm::alloc::device:
            case sycl::usm::alloc::shared: return location::device;
            case sycl::usm::alloc::host: return location::host;
            default: return location::unknown;
        }
    }
    catch (...) {
        return location::unknown;
    }
}

void accessor::copy_device_to_host(void* destination,
                                   const void* source,
                                   std::size_t bytes) const {
    if (bytes == 0) {
        return;
    }
    CCL_THROW_IF_NOT(sycl_queue,
                     "a device buffer was passed without a stream; device operands "
                     "require a ccl::stream so the backend has a queue to copy with");
    sycl_queue->memcpy(destination, source, bytes).wait();
}

void accessor::copy_host_to_device(void* destination,
                                   const void* source,
                                   std::size_t bytes) const {
    if (bytes == 0) {
        return;
    }
    CCL_THROW_IF_NOT(sycl_queue,
                     "a device buffer was passed without a stream; device operands "
                     "require a ccl::stream so the backend has a queue to copy with");
    sycl_queue->memcpy(destination, source, bytes).wait();
}

void accessor::synchronize() const {
    if (sycl_queue) {
        sycl_queue->wait();
    }
}

#else // CCL_ENABLE_SYCL

/* Without SYCL there is no way to recognise or reach device memory, so every
 * buffer is host memory and the copy helpers are unreachable by construction. */

bool accessor::enabled() const noexcept {
    return false;
}

location accessor::classify(const void*) const noexcept {
    return location::host;
}

namespace {

[[noreturn]] void unavailable(const char* operation) {
    CCL_THROW(operation,
              " requires a SYCL-enabled build; rebuild oneCCL with CCL_ENABLE_SYCL=ON");
}

} // namespace

void accessor::copy_device_to_host(void*, const void*, std::size_t) const {
    unavailable("device to host copy");
}

void accessor::copy_host_to_device(void*, const void*, std::size_t) const {
    unavailable("host to device copy");
}

void accessor::synchronize() const {}

#endif // CCL_ENABLE_SYCL

#ifdef CCL_ENABLE_OSHMPI_PINNED_STAGING

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

#else // CCL_ENABLE_OSHMPI_PINNED_STAGING

/* Not an error: the arena works unregistered, and this is the default build. */

bool try_register_host_memory(void*, std::size_t) noexcept {
    return false;
}

void unregister_host_memory(void*) noexcept {}

#endif // CCL_ENABLE_OSHMPI_PINNED_STAGING

} // namespace oshmpi_device
} // namespace ccl

#endif // CCL_ENABLE_OSHMPI
