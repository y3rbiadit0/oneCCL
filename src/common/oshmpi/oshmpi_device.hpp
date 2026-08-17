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
#pragma once

// CCL_ENABLE_SYCL comes from the generated config header, not the command line,
// so include it before testing it.
#include "oneapi/ccl/config.h"

#ifdef CCL_ENABLE_OSHMPI

#include <cstddef>

#ifdef CCL_ENABLE_SYCL
#include <optional>
#include <sycl/sycl.hpp>
#endif // CCL_ENABLE_SYCL

namespace ccl {
namespace oshmpi_device {

// `unknown` is treated as host memory: that is what an ordinary malloc reports.
enum class location { host, device, unknown };

/* What the staging path needs to touch device memory: a context to classify
 * pointers against, and a queue to copy with. A default constructed accessor is
 * host-only - every buffer classifies as host and staging falls back to memcpy. */
class accessor {
public:
    accessor() = default;

#ifdef CCL_ENABLE_SYCL
    // A context alone can classify; the queue comes from the caller's stream and
    // may be absent, in which case a device operand is an error rather than a
    // silent host memcpy over device memory.
    explicit accessor(sycl::context context) : sycl_context(std::move(context)) {}

    accessor(sycl::context context, sycl::queue queue)
            : sycl_context(std::move(context)),
              sycl_queue(std::move(queue)) {}
#endif // CCL_ENABLE_SYCL

    bool enabled() const noexcept;

    location classify(const void* buffer) const noexcept;

    bool is_device(const void* buffer) const noexcept {
        return classify(buffer) == location::device;
    }

    // Synchronous: OSHMPI collectives block, so there is no stream to order
    // against. Throw when a device operand arrives without a queue.
    void copy_device_to_host(void* destination, const void* source, std::size_t bytes) const;
    void copy_host_to_device(void* destination, const void* source, std::size_t bytes) const;

    // Needed before reading a buffer the caller may have just written from a kernel.
    void synchronize() const;

private:
#ifdef CCL_ENABLE_SYCL
    std::optional<sycl::context> sycl_context;
    // Mutable: queue::memcpy and wait are non-const, but the queue is a
    // reference-counted handle to the caller's queue, not state we own.
    mutable std::optional<sycl::queue> sycl_queue;
#endif // CCL_ENABLE_SYCL
};

/* Optional accelerator, nothing depends on it: pinning only removes the driver's
 * bounce buffer from staged copies. CUDA-only because page-locking memory you did
 * not allocate has no SYCL equivalent (SYCLomatic DPCT1027), and the arena must
 * come from shmem_malloc to be symmetric. Guarded by
 * CCL_ENABLE_OSHMPI_PINNED_STAGING, off by default.
 *
 * Best effort: reports failure rather than throwing. Returns true when the pages
 * were pinned and must later be released with unregister_host_memory. */
bool try_register_host_memory(void* buffer, std::size_t bytes) noexcept;
void unregister_host_memory(void* buffer) noexcept;

} // namespace oshmpi_device
} // namespace ccl

#endif // CCL_ENABLE_OSHMPI
