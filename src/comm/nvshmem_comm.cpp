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

#include "comm/nvshmem_comm.hpp"

#ifdef CCL_ENABLE_NVSHMEM

#include <algorithm>
#include <limits>

namespace ccl {

namespace {

#define NVSHMEM_M1_UNSUPPORTED(operation) \
    CCL_THROW(#operation " is not supported by the NVSHMEM M1 backend")

} // namespace

nvshmem_comm::nvshmem_comm(device_t device,
                           context_t context,
                           size_t size,
                           size_t rank,
                           std::shared_ptr<ccl::kvs_interface> kvs)
        : device_ptr(std::make_shared<ccl::device>(device)),
          context_ptr(std::make_shared<ccl::context>(context)),
          kvs(std::move(kvs)),
          comm_rank(static_cast<int>(rank)),
          comm_size(static_cast<int>(size)) {
    CCL_THROW_IF_NOT(size > 0, "NVSHMEM communicator size must be positive");
    CCL_THROW_IF_NOT(rank < size, "NVSHMEM communicator rank is out of range");
    CCL_THROW_IF_NOT(size <= static_cast<size_t>(std::numeric_limits<int>::max()),
                     "NVSHMEM communicator size exceeds the supported range");
    CCL_THROW_IF_NOT(this->kvs, "NVSHMEM communicator requires a KVS instance");

    const auto& native_device = device_ptr->get_native();
    const auto& native_context = context_ptr->get_native();
    CCL_THROW_IF_NOT(native_device.get_backend() == sycl::backend::ext_oneapi_cuda,
                     "NVSHMEM backend requires a CUDA-backed SYCL device");
    CCL_THROW_IF_NOT(native_context.get_backend() == sycl::backend::ext_oneapi_cuda,
                     "NVSHMEM backend requires a CUDA-backed SYCL context");

    const auto context_devices = native_context.get_devices();
    CCL_THROW_IF_NOT(std::find(context_devices.begin(), context_devices.end(), native_device) !=
                         context_devices.end(),
                     "NVSHMEM SYCL context does not contain the selected device");

    internal_queue = std::make_shared<sycl::queue>(
        native_context, native_device, sycl::property::queue::in_order{});
}

nvshmem_comm* nvshmem_comm::create(device_t device,
                                   context_t context,
                                   size_t size,
                                   size_t rank,
                                   std::shared_ptr<ccl::kvs_interface> kvs) {
    CCL_THROW_IF_NOT(size > 0, "NVSHMEM communicator size must be positive");
    CCL_THROW_IF_NOT(rank < size, "NVSHMEM communicator rank is out of range");
    CCL_THROW_IF_NOT(size <= static_cast<size_t>(std::numeric_limits<int>::max()),
                     "NVSHMEM communicator size exceeds the supported range");
    return new nvshmem_comm(device, context, size, rank, std::move(kvs));
}

ccl::comm_interface_ptr nvshmem_comm::split(int color, int key, bool split_external_use) {
    NVSHMEM_M1_UNSUPPORTED(split);
}

ccl::event nvshmem_comm::barrier(const ccl::stream::impl_value_t& stream,
                                 const ccl::barrier_attr& attr,
                                 const ccl::vector_class<ccl::event>& deps) {
    NVSHMEM_M1_UNSUPPORTED(barrier);
}

ccl::event nvshmem_comm::allgather_impl(const void* send_buf,
                                        void* recv_buf,
                                        size_t count,
                                        ccl::datatype dtype,
                                        const ccl::stream::impl_value_t& stream,
                                        const ccl::allgather_attr& attr,
                                        const ccl::vector_class<ccl::event>& deps) {
    NVSHMEM_M1_UNSUPPORTED(allgather);
}

ccl::event nvshmem_comm::allgather_impl(const void* send_buf,
                                        const ccl::vector_class<void*>& recv_buf,
                                        size_t count,
                                        ccl::datatype dtype,
                                        const ccl::stream::impl_value_t& stream,
                                        const ccl::allgather_attr& attr,
                                        const ccl::vector_class<ccl::event>& deps) {
    NVSHMEM_M1_UNSUPPORTED(allgather);
}

ccl::event nvshmem_comm::allgatherv_impl(
    const void* send_buf,
    size_t send_count,
    void* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::datatype dtype,
    const ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    NVSHMEM_M1_UNSUPPORTED(allgatherv);
}

ccl::event nvshmem_comm::allgatherv_impl(
    const void* send_buf,
    size_t send_count,
    const ccl::vector_class<void*>& recv_bufs,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::datatype dtype,
    const ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    NVSHMEM_M1_UNSUPPORTED(allgatherv);
}

ccl::event nvshmem_comm::allreduce_impl(const void* send_buf,
                                        void* recv_buf,
                                        size_t count,
                                        ccl::datatype dtype,
                                        ccl::reduction reduction,
                                        const ccl::stream::impl_value_t& stream,
                                        const ccl::allreduce_attr& attr,
                                        const ccl::vector_class<ccl::event>& deps) {
    NVSHMEM_M1_UNSUPPORTED(allreduce);
}

ccl::event nvshmem_comm::alltoall_impl(const void* send_buf,
                                       void* recv_buf,
                                       size_t count,
                                       ccl::datatype dtype,
                                       const ccl::stream::impl_value_t& stream,
                                       const ccl::alltoall_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    NVSHMEM_M1_UNSUPPORTED(alltoall);
}

ccl::event nvshmem_comm::alltoallv_impl(
    const void* send_buf,
    const ccl::vector_class<size_t>& send_counts,
    void* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::datatype dtype,
    const ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    NVSHMEM_M1_UNSUPPORTED(alltoallv);
}

ccl::event nvshmem_comm::broadcast_impl(void* buf,
                                        size_t count,
                                        ccl::datatype dtype,
                                        int root,
                                        const ccl::stream::impl_value_t& stream,
                                        const ccl::broadcast_attr& attr,
                                        const ccl::vector_class<ccl::event>& deps) {
    NVSHMEM_M1_UNSUPPORTED(broadcast);
}

ccl::event nvshmem_comm::broadcast_impl(void* send_buf,
                                        void* recv_buf,
                                        size_t count,
                                        ccl::datatype dtype,
                                        int root,
                                        const ccl::stream::impl_value_t& stream,
                                        const ccl::broadcast_attr& attr,
                                        const ccl::vector_class<ccl::event>& deps) {
    NVSHMEM_M1_UNSUPPORTED(broadcast);
}

ccl::event nvshmem_comm::reduce_impl(const void* send_buf,
                                     void* recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     ccl::reduction reduction,
                                     int root,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::reduce_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    NVSHMEM_M1_UNSUPPORTED(reduce);
}

ccl::event nvshmem_comm::reduce_scatter_impl(
    const void* send_buf,
    void* recv_buf,
    size_t recv_count,
    ccl::datatype dtype,
    ccl::reduction reduction,
    const ccl::stream::impl_value_t& stream,
    const ccl::reduce_scatter_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    NVSHMEM_M1_UNSUPPORTED(reduce_scatter);
}

ccl::event nvshmem_comm::recv_impl(void* recv_buf,
                                   size_t recv_count,
                                   ccl::datatype dtype,
                                   int peer,
                                   const ccl::stream::impl_value_t& stream,
                                   const ccl::pt2pt_attr& attr,
                                   const ccl::vector_class<ccl::event>& deps) {
    NVSHMEM_M1_UNSUPPORTED(recv);
}

ccl::event nvshmem_comm::send_impl(void* send_buf,
                                   size_t send_count,
                                   ccl::datatype dtype,
                                   int peer,
                                   const ccl::stream::impl_value_t& stream,
                                   const ccl::pt2pt_attr& attr,
                                   const ccl::vector_class<ccl::event>& deps) {
    NVSHMEM_M1_UNSUPPORTED(send);
}

#undef NVSHMEM_M1_UNSUPPORTED

} // namespace ccl

#endif // CCL_ENABLE_NVSHMEM
