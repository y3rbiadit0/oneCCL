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

#include "oneapi/ccl/type_traits.hpp"

namespace ccl {
namespace detail {

template <class buffer_type>
ccl::vector_class<void*> erase_buffer_types(const ccl::vector_class<buffer_type*>& buffers) {
    ccl::vector_class<void*> result;
    result.reserve(buffers.size());
    for (auto* buffer : buffers) {
        result.push_back(static_cast<void*>(buffer));
    }
    return result;
}

} // namespace detail

template <class buffer_type>
ccl::event oshmpi_comm::allgather_impl(const buffer_type* send_buf,
                                       buffer_type* recv_buf,
                                       std::size_t count,
                                       const ccl::stream::impl_value_t& stream,
                                       const ccl::allgather_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    return allgather_impl(static_cast<const void*>(send_buf),
                          static_cast<void*>(recv_buf),
                          count,
                          ccl::native_type_info<buffer_type>::dtype,
                          stream,
                          attr,
                          deps);
}

template <class buffer_type>
ccl::event oshmpi_comm::allgather_impl(const buffer_type* send_buf,
                                       ccl::vector_class<buffer_type*>& recv_buf,
                                       std::size_t count,
                                       const ccl::stream::impl_value_t& stream,
                                       const ccl::allgather_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    return allgather_impl(static_cast<const void*>(send_buf),
                          detail::erase_buffer_types(recv_buf),
                          count,
                          ccl::native_type_info<buffer_type>::dtype,
                          stream,
                          attr,
                          deps);
}

template <class buffer_type>
ccl::event oshmpi_comm::allgatherv_impl(const buffer_type* send_buf,
                                        std::size_t send_count,
                                        buffer_type* recv_buf,
                                        const ccl::vector_class<std::size_t>& recv_counts,
                                        const ccl::stream::impl_value_t& stream,
                                        const ccl::allgatherv_attr& attr,
                                        const ccl::vector_class<ccl::event>& deps) {
    return allgatherv_impl(static_cast<const void*>(send_buf),
                           send_count,
                           static_cast<void*>(recv_buf),
                           recv_counts,
                           ccl::native_type_info<buffer_type>::dtype,
                           stream,
                           attr,
                           deps);
}

template <class buffer_type>
ccl::event oshmpi_comm::allgatherv_impl(const buffer_type* send_buf,
                                        std::size_t send_count,
                                        ccl::vector_class<buffer_type*>& recv_buf,
                                        const ccl::vector_class<std::size_t>& recv_counts,
                                        const ccl::stream::impl_value_t& stream,
                                        const ccl::allgatherv_attr& attr,
                                        const ccl::vector_class<ccl::event>& deps) {
    return allgatherv_impl(static_cast<const void*>(send_buf),
                           send_count,
                           detail::erase_buffer_types(recv_buf),
                           recv_counts,
                           ccl::native_type_info<buffer_type>::dtype,
                           stream,
                           attr,
                           deps);
}

template <class buffer_type>
ccl::event oshmpi_comm::allreduce_impl(const buffer_type* send_buf,
                                       buffer_type* recv_buf,
                                       std::size_t count,
                                       ccl::reduction reduction,
                                       const ccl::stream::impl_value_t& stream,
                                       const ccl::allreduce_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    return allreduce_impl(static_cast<const void*>(send_buf),
                          static_cast<void*>(recv_buf),
                          count,
                          ccl::native_type_info<buffer_type>::dtype,
                          reduction,
                          stream,
                          attr,
                          deps);
}

template <class buffer_type>
ccl::event oshmpi_comm::alltoall_impl(const buffer_type* send_buf,
                                      buffer_type* recv_buf,
                                      std::size_t count,
                                      const ccl::stream::impl_value_t& stream,
                                      const ccl::alltoall_attr& attr,
                                      const ccl::vector_class<ccl::event>& deps) {
    return alltoall_impl(static_cast<const void*>(send_buf),
                         static_cast<void*>(recv_buf),
                         count,
                         ccl::native_type_info<buffer_type>::dtype,
                         stream,
                         attr,
                         deps);
}

template <class buffer_type>
ccl::event oshmpi_comm::alltoall_impl(const ccl::vector_class<buffer_type*>& send_buf,
                                      const ccl::vector_class<buffer_type*>& recv_buf,
                                      std::size_t count,
                                      const ccl::stream::impl_value_t& stream,
                                      const ccl::alltoall_attr& attr,
                                      const ccl::vector_class<ccl::event>& deps) {
    return alltoall_impl(detail::erase_buffer_types(send_buf),
                         detail::erase_buffer_types(recv_buf),
                         count,
                         ccl::native_type_info<buffer_type>::dtype,
                         stream,
                         attr,
                         deps);
}

template <class buffer_type>
ccl::event oshmpi_comm::alltoallv_impl(const buffer_type* send_buf,
                                       const ccl::vector_class<std::size_t>& send_counts,
                                       buffer_type* recv_buf,
                                       const ccl::vector_class<std::size_t>& recv_counts,
                                       const ccl::stream::impl_value_t& stream,
                                       const ccl::alltoallv_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    return alltoallv_impl(static_cast<const void*>(send_buf),
                          send_counts,
                          static_cast<void*>(recv_buf),
                          recv_counts,
                          ccl::native_type_info<buffer_type>::dtype,
                          stream,
                          attr,
                          deps);
}

template <class buffer_type>
ccl::event oshmpi_comm::alltoallv_impl(const ccl::vector_class<buffer_type*>& send_buf,
                                       const ccl::vector_class<std::size_t>& send_counts,
                                       const ccl::vector_class<buffer_type*>& recv_buf,
                                       const ccl::vector_class<std::size_t>& recv_counts,
                                       const ccl::stream::impl_value_t& stream,
                                       const ccl::alltoallv_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    return alltoallv_impl(detail::erase_buffer_types(send_buf),
                          send_counts,
                          detail::erase_buffer_types(recv_buf),
                          recv_counts,
                          ccl::native_type_info<buffer_type>::dtype,
                          stream,
                          attr,
                          deps);
}

template <class buffer_type>
ccl::event oshmpi_comm::broadcast_impl(buffer_type* buf,
                                       std::size_t count,
                                       int root,
                                       const ccl::stream::impl_value_t& stream,
                                       const ccl::broadcast_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    return broadcast_impl(static_cast<void*>(buf),
                          count,
                          ccl::native_type_info<buffer_type>::dtype,
                          root,
                          stream,
                          attr,
                          deps);
}

template <class buffer_type>
ccl::event oshmpi_comm::broadcast_impl(buffer_type* send_buf,
                                       buffer_type* recv_buf,
                                       std::size_t count,
                                       int root,
                                       const ccl::stream::impl_value_t& stream,
                                       const ccl::broadcast_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    return broadcast_impl(static_cast<void*>(send_buf),
                          static_cast<void*>(recv_buf),
                          count,
                          ccl::native_type_info<buffer_type>::dtype,
                          root,
                          stream,
                          attr,
                          deps);
}

template <class buffer_type>
ccl::event oshmpi_comm::reduce_impl(const buffer_type* send_buf,
                                    buffer_type* recv_buf,
                                    std::size_t count,
                                    ccl::reduction reduction,
                                    int root,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::reduce_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    return reduce_impl(static_cast<const void*>(send_buf),
                       static_cast<void*>(recv_buf),
                       count,
                       ccl::native_type_info<buffer_type>::dtype,
                       reduction,
                       root,
                       stream,
                       attr,
                       deps);
}

template <class buffer_type>
ccl::event oshmpi_comm::reduce_scatter_impl(const buffer_type* send_buf,
                                            buffer_type* recv_buf,
                                            std::size_t recv_count,
                                            ccl::reduction reduction,
                                            const ccl::stream::impl_value_t& stream,
                                            const ccl::reduce_scatter_attr& attr,
                                            const ccl::vector_class<ccl::event>& deps) {
    return reduce_scatter_impl(static_cast<const void*>(send_buf),
                               static_cast<void*>(recv_buf),
                               recv_count,
                               ccl::native_type_info<buffer_type>::dtype,
                               reduction,
                               stream,
                               attr,
                               deps);
}

template <class buffer_type>
ccl::event oshmpi_comm::recv_impl(buffer_type* recv_buf,
                                  std::size_t recv_count,
                                  int peer,
                                  const ccl::stream::impl_value_t& stream,
                                  const ccl::pt2pt_attr& attr,
                                  const ccl::vector_class<ccl::event>& deps) {
    return recv_impl(static_cast<void*>(recv_buf),
                     recv_count,
                     ccl::native_type_info<buffer_type>::dtype,
                     peer,
                     stream,
                     attr,
                     deps);
}

template <class buffer_type>
ccl::event oshmpi_comm::send_impl(buffer_type* send_buf,
                                  std::size_t send_count,
                                  int peer,
                                  const ccl::stream::impl_value_t& stream,
                                  const ccl::pt2pt_attr& attr,
                                  const ccl::vector_class<ccl::event>& deps) {
    return send_impl(static_cast<void*>(send_buf),
                     send_count,
                     ccl::native_type_info<buffer_type>::dtype,
                     peer,
                     stream,
                     attr,
                     deps);
}

} // namespace ccl
