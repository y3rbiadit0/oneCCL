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

/* CCL_ENABLE_SYCL comes from the generated config header rather than the compiler
 * command line, so include it before testing it - otherwise a translation unit
 * that includes this header first declares the host-only class and then compiles
 * the SYCL definitions against it. */
#include "oneapi/ccl/config.h"

#ifdef CCL_ENABLE_OSHMPI

#include <cstddef>

#ifdef CCL_ENABLE_SYCL
#include <optional>
#include <sycl/sycl.hpp>
#endif // CCL_ENABLE_SYCL

namespace ccl {
namespace oshmpi_device {

/* Where a caller's buffer lives. `unknown` covers pointers the SYCL runtime does
 * not recognise, which are treated as host memory - that is what an ordinary
 * malloc reports, and it routes the buffer down the path that cannot corrupt
 * device memory. */
enum class location { host, device, unknown };

/* Everything the staging path needs in order to touch device memory: a context to
 * classify pointers against, and a queue to copy with.
 *
 * SYCL throughout, with no CUDA anywhere: classification, copying and
 * synchronisation all have direct SYCL equivalents, so the backend works on any
 * target the SYCL implementation supports rather than only on NVIDIA hardware.
 *
 * A default constructed accessor is host-only. That is what a host communicator
 * gets, and what every communicator gets in a build without SYCL: it reports every
 * buffer as host, so the staging path falls back to plain memcpy exactly as it did
 * before device support existed. */
class accessor {
public:
    accessor() = default;

#ifdef CCL_ENABLE_SYCL
    /* A context alone is enough to classify. The queue is separate because it
     * comes from the caller's stream, which is absent on some call paths - a
     * device buffer discovered without a queue is an error rather than a silent
     * host memcpy over device memory. */
    explicit accessor(sycl::context context) : sycl_context(std::move(context)) {}

    accessor(sycl::context context, sycl::queue queue)
            : sycl_context(std::move(context)),
              sycl_queue(std::move(queue)) {}
#endif // CCL_ENABLE_SYCL

    /* True when this accessor can recognise device memory at all. */
    bool enabled() const noexcept;

    location classify(const void* buffer) const noexcept;

    bool is_device(const void* buffer) const noexcept {
        return classify(buffer) == location::device;
    }

    /* Synchronous copies. OSHMPI collectives block and return an already completed
     * event, so there is no stream to order against and no benefit to the async
     * variants. Throw when a device operand arrives without a queue to copy it. */
    void copy_device_to_host(void* destination, const void* source, std::size_t bytes) const;
    void copy_host_to_device(void* destination, const void* source, std::size_t bytes) const;

    /* Blocks until previously issued device work has completed. Needed before a
     * collective reads a buffer the caller may have just written from a kernel. */
    void synchronize() const;

private:
#ifdef CCL_ENABLE_SYCL
    std::optional<sycl::context> sycl_context;
    /* Mutable because sycl::queue::memcpy and wait are non-const, while copying
     * through a queue does not change this accessor's own state: the queue is a
     * reference-counted handle to the caller's queue, not state we own. */
    mutable std::optional<sycl::queue> sycl_queue;
#endif // CCL_ENABLE_SYCL
};

/* Page-locking the staging arena, an optional accelerator rather than part of the
 * device path. Nothing above depends on it and the backend is fully functional
 * without it - it only removes the driver's bounce buffer from staged copies.
 *
 * It is CUDA-only because there is no portable way to page-lock memory you did
 * not allocate: SYCLomatic's DPCT1027 lists cuMemHostRegister among the calls it
 * replaces with 0 for want of a SYCL equivalent, and the arena has to come from
 * shmem_malloc to be symmetric, so sycl::malloc_host is not an option either.
 * Guarded by CCL_ENABLE_OSHMPI_PINNED_STAGING, off by default; measured worth
 * ~45% of peak bandwidth on Leonardo, and the SYCL copy path does honour the
 * registration.
 *
 * Registration is best effort and reports failure rather than throwing - the
 * pages may already be registered, or the build may have no CUDA at all. Returns
 * true when the pages were pinned and must later be released with
 * unregister_host_memory. */
bool try_register_host_memory(void* buffer, std::size_t bytes) noexcept;
void unregister_host_memory(void* buffer) noexcept;

} // namespace oshmpi_device
} // namespace ccl

#endif // CCL_ENABLE_OSHMPI
