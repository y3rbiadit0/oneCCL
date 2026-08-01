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

#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace ccl {
namespace v1 {
class kvs_interface;
}
namespace nvshmem {

class runtime {
public:
    using staged_operation =
        std::function<int(void* destination,
                          const void* source,
                          size_t bytes,
                          void* native_stream)>;

    runtime() = default;
    runtime(const runtime&) = delete;
    runtime(runtime&&) = delete;
    runtime& operator=(const runtime&) = delete;
    runtime& operator=(runtime&&) = delete;
    ~runtime() noexcept;

    void initialize(const sycl::device& device,
                    const sycl::context& context,
                    int rank,
                    int size,
                    const std::shared_ptr<ccl::v1::kvs_interface>& kvs);
    void release() noexcept;

    sycl::event submit_staged(const void* source,
                              void* destination,
                              size_t bytes,
                              sycl::queue& queue,
                              const staged_operation& operation);
    void validate_staging(sycl::queue& queue, bool enabled);

    size_t staging_lane_size() const;
    bool is_ready() const;

private:
    enum class state { uninitialized, initializing, ready, failed, finalized };

    void initialize_locked(const sycl::device& device,
                           const sycl::context& context,
                           int rank,
                           int size,
                           const std::shared_ptr<ccl::v1::kvs_interface>& kvs);
    void allocate_arena_locked();
    void finalize_locked() noexcept;
    void validate_existing_locked(const sycl::device& device,
                                  const sycl::context& context,
                                  int rank,
                                  int size) const;

    mutable std::mutex mutex;
    state current_state{ state::uninitialized };
    bool owns_initialization{ false };
    int pe_rank{ -1 };
    int pe_count{ 0 };
    int cuda_device{ -1 };
    std::optional<sycl::device> device;
    std::optional<sycl::context> context;
    void* arena{ nullptr };
    void* source_lane{ nullptr };
    void* destination_lane{ nullptr };
    void* metadata_region{ nullptr };
    void* signal_region{ nullptr };
    size_t arena_bytes{ 0 };
    size_t lane_bytes{ 0 };
    size_t communicator_count{ 0 };
    std::optional<sycl::event> last_completion;
    std::shared_ptr<ccl::v1::kvs_interface> bootstrap_kvs;
    std::string key_prefix;
    bool staging_validation_completed{ false };
};

} // namespace nvshmem
} // namespace ccl
