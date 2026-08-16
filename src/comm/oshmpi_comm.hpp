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

#include "comm/comm_interface.hpp"

namespace ccl {

class oshmpi_comm final : public ccl::comm_interface {
public:
    oshmpi_comm(const oshmpi_comm&) = delete;
    oshmpi_comm(oshmpi_comm&&) = delete;
    oshmpi_comm& operator=(const oshmpi_comm&) = delete;
    oshmpi_comm& operator=(oshmpi_comm&&) = delete;
    ~oshmpi_comm() override;

    static oshmpi_comm* create(std::size_t size,
                               std::size_t rank,
                               std::shared_ptr<ccl::kvs_interface> kvs_interface);

    /* Device overload. The backend does not dispatch on the device or context -
     * buffers are classified by pointer and the collectives are the same either
     * way - but a device communicator must report them back, and holding them
     * keeps the caller's device and context alive for the communicator's
     * lifetime. */
    static oshmpi_comm* create(device_t device,
                               context_t context,
                               std::size_t size,
                               std::size_t rank,
                               std::shared_ptr<ccl::kvs_interface> kvs_interface);

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

    ccl::comm_interface_ptr split(int color, int key, bool split_external_use) override {
        CCL_THROW("split is not supported for OSHMPI backend");
    }

    ccl::event barrier(const ccl::stream::impl_value_t& stream,
                       const ccl::barrier_attr& attr,
                       const ccl::vector_class<ccl::event>& deps = {}) override {
        return barrier_impl(stream, attr, deps);
    }

    ccl::event barrier_impl(const ccl::stream::impl_value_t& stream,
                            const ccl::barrier_attr& attr,
                            const ccl::vector_class<ccl::event>& deps = {});

    COMM_INTERFACE_COLL_METHODS(DEFINITION);
    COMM_IMPL_DECLARATION;

private:
    oshmpi_comm(std::size_t size,
                std::size_t rank,
                std::shared_ptr<ccl::kvs> kvs,
                device_ptr_t device = {},
                context_ptr_t context = {});

    oshmpi_comm* get_impl() {
        return this;
    }

    int comm_rank;
    int comm_size;
    std::shared_ptr<ccl::kvs> kvs;
    // Empty for a host communicator.
    device_ptr_t device_ptr;
    context_ptr_t context_ptr;
};

} // namespace ccl

#include "comm/oshmpi_comm_impl.hpp"

#endif // CCL_ENABLE_OSHMPI
