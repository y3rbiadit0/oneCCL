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

#ifdef CCL_ENABLE_NVSHMEM

#include "comm/comm_interface.hpp"

#if !defined(CCL_ENABLE_SYCL)
#error "The NVSHMEM backend requires SYCL support"
#endif

#include <sycl/sycl.hpp>

namespace ccl {

class alignas(CACHELINE_SIZE) nvshmem_comm : public ccl::comm_interface {
public:
    nvshmem_comm(nvshmem_comm& src) = delete;
    nvshmem_comm(nvshmem_comm&& src) = delete;
    nvshmem_comm& operator=(nvshmem_comm& src) = delete;
    nvshmem_comm& operator=(nvshmem_comm&& src) = delete;
    ~nvshmem_comm() override = default;

    static nvshmem_comm* create(device_t device,
                                context_t context,
                                size_t size,
                                size_t rank,
                                std::shared_ptr<ccl::kvs_interface> kvs);

    int rank() const override {
        return comm_rank;
    }

    int size() const override {
        return comm_size;
    }

    device_ptr_t get_device() const override {
        return device_ptr;
    }

    context_ptr_t get_context() const override {
        return context_ptr;
    }

    ccl::comm_interface_ptr split(int color, int key, bool split_external_use) override;

    ccl::event barrier(const ccl::stream::impl_value_t& stream,
                       const ccl::barrier_attr& attr,
                       const ccl::vector_class<ccl::event>& deps = {}) override;

    COMM_INTERFACE_COLL_DEFINITION__VOID_REQUIRED

    COMM_IMPL_DECLARATION_VOID_REQUIRED

private:
    nvshmem_comm(device_t device,
                 context_t context,
                 size_t size,
                 size_t rank,
                 std::shared_ptr<ccl::kvs_interface> kvs);

    nvshmem_comm* get_impl() {
        return this;
    }

    device_ptr_t device_ptr;
    context_ptr_t context_ptr;
    std::shared_ptr<sycl::queue> internal_queue;
    std::shared_ptr<ccl::kvs_interface> kvs;
    int comm_rank;
    int comm_size;
};

} // namespace ccl

#endif // CCL_ENABLE_NVSHMEM
