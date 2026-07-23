/*
 Copyright 2016-2020 Intel Corporation

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

#ifdef CCL_ENABLE_NCCL

#include "comm/comm_interface.hpp"
#include "common/api_wrapper/nccl_api_wrapper.hpp"
#include <cuda_runtime.h>

namespace ccl {

class nccl_kvs_impl;

class alignas(CACHELINE_SIZE) nccl_comm : public ccl::comm_interface {
public:
    nccl_comm(nccl_comm& src) = delete;
    nccl_comm(nccl_comm&& src) = default;
    nccl_comm& operator=(nccl_comm& src) = delete;
    nccl_comm& operator=(nccl_comm&& src) = default;
    ~nccl_comm();

    static nccl_comm* create(device_t device,
                             context_t context,
                             size_t size,
                             size_t rank,
                             std::shared_ptr<ccl::kvs_interface> kvs_interface);

private:
    nccl_comm(device_t device,
              context_t context,
              size_t size,
              size_t rank,
              ncclUniqueId nccl_id,
              std::shared_ptr<ccl::kvs> kvs,
              const ccl::nccl_kvs_impl* kvs_impl);

public:
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

    ccl::event make_event(const ccl::stream::impl_value_t& stream);

    static void group_start();
    static void group_end();
    static bool is_group_active();

    // collective operation declarations
    ccl::event barrier(const ccl::stream::impl_value_t& stream,
                       const ccl::barrier_attr& attr,
                       const ccl::vector_class<ccl::event>& deps = {}) override {
        return barrier_impl(stream, attr, deps);
    }

    ccl::event barrier_impl(const ccl::stream::impl_value_t& stream,
                            const ccl::barrier_attr& attr,
                            const ccl::vector_class<ccl::event>& deps = {});

    COMM_INTERFACE_COLL_DEFINITION__VOID_REQUIRED

    COMM_IMPL_DECLARATION_VOID_REQUIRED

    ccl::comm_interface_ptr split(int color, int key, bool split_external_use) override {
        CCL_THROW("split is not supported for NCCL backend yet");
    }

private:
    void register_group_operation(const ccl::stream::impl_value_t& stream);
    static ccl::event make_group_event(const ccl::stream::impl_value_t& stream);
    ccl::event make_operation_event(const ccl::stream::impl_value_t& stream);

    nccl_comm* get_impl() {
        return this;
    }

    cudaStream_t get_cuda_stream(const ccl::stream::impl_value_t& stream);
    ncclDataType_t get_nccl_datatype(ccl::datatype dtype);
    ncclRedOp_t get_nccl_reduction(ccl::reduction reduction);

    device_ptr_t device_ptr;
    context_ptr_t context_ptr;

    size_t comm_rank;
    size_t comm_size;

    ncclComm_t nccl_comm_handle = nullptr;

    // while we only use the impl, keep the original object to avoid early destruction
    std::shared_ptr<ccl::kvs> kvs;
    const ccl::nccl_kvs_impl* kvs_impl;
};

} // namespace ccl

#endif // CCL_ENABLE_NCCL
