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

#ifdef CCL_ENABLE_OSHMPI

#include <cstddef>

namespace ccl {
namespace oshmpi_device {

/* Where a caller's buffer lives. Phase 2 accepts device pointers through the
 * ordinary void* API rather than through a SYCL stream, so the backend has to
 * classify them itself. `unknown` covers pointers CUDA does not recognise, which
 * are treated as host memory - that is what an unregistered host allocation
 * reports. */
enum class location { host, device, unknown };

/* True when this build can classify and copy device memory at all. When false
 * every buffer is reported as host and the device copy helpers must not be
 * called. */
bool enabled() noexcept;

/* Never throws: a classification failure means "not device memory", which is
 * the safe answer - it routes the buffer down the host path that Phase 1 already
 * validates. */
location classify(const void* buffer) noexcept;

inline bool is_device(const void* buffer) noexcept {
    return classify(buffer) == location::device;
}

/* Synchronous copies. Phase 1 semantics are blocking and return an already
 * completed event, so there is no stream to order against and no benefit to the
 * async variants. */
void copy_device_to_host(void* destination, const void* source, std::size_t bytes);
void copy_host_to_device(void* destination, const void* source, std::size_t bytes);
void copy_device_to_device(void* destination, const void* source, std::size_t bytes);

/* Blocks until previously issued device work has completed. Needed before a
 * collective reads a staging buffer the caller may have just written. */
void synchronize();

} // namespace oshmpi_device
} // namespace ccl

#endif // CCL_ENABLE_OSHMPI
