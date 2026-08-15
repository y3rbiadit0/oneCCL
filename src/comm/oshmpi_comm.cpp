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

#include "comm/oshmpi_comm.hpp"

#include <limits>

#include "oneapi/ccl/api_functions.hpp"
#include "common/datatype/datatype.hpp"
#include "common/oshmpi/oshmpi_runtime.hpp"
#include "oshmpi_kvs_impl.hpp"

namespace ccl {
namespace {

void wait_dependencies(const ccl::vector_class<ccl::event>& deps) {
    for (const auto& dependency : deps) {
        const_cast<ccl::event&>(dependency).wait();
    }
}

template <class attr_type>
void validate_attributes(const attr_type& attr) {
    CCL_THROW_IF_NOT(attr.template get<ccl::operation_attr_id::priority>() == 0,
                     "OSHMPI backend does not support operation priority");
    CCL_THROW_IF_NOT(!attr.template get<ccl::operation_attr_id::to_cache>(),
                     "OSHMPI backend does not support operation caching");
    // ccl::string exposes length(), not empty() - see coll_param.cpp
    CCL_THROW_IF_NOT(attr.template get<ccl::operation_attr_id::match_id>().length() == 0,
                     "OSHMPI backend does not support match identifiers");
}

std::size_t checked_bytes(std::size_t count, ccl::datatype dtype) {
    const std::size_t datatype_size = ccl::get_datatype_size(dtype);
    CCL_THROW_IF_NOT(count == 0 ||
                         datatype_size <= std::numeric_limits<std::size_t>::max() / count,
                     "collective byte count overflow");
    return count * datatype_size;
}

void validate_buffer(const void* buffer, std::size_t count, const char* name) {
    CCL_THROW_IF_NOT(count == 0 || buffer, name, " must be non-null for a non-zero count");
}

// Not named `unsupported`: ccl::unsupported is an exception type exported by
// oneapi/ccl/exception.hpp, and an unqualified call from inside namespace ccl
// would be ambiguous against it.
[[noreturn]] void throw_unsupported(const char* operation) {
    CCL_THROW(operation, " is not supported for OSHMPI backend");
}

} // namespace

oshmpi_comm::oshmpi_comm(std::size_t size, std::size_t rank, std::shared_ptr<ccl::kvs> kvs)
        : comm_rank(static_cast<int>(rank)), comm_size(static_cast<int>(size)), kvs(std::move(kvs)) {
    oshmpi_runtime::instance().acquire(size, rank);
}

oshmpi_comm::~oshmpi_comm() {
    oshmpi_runtime::instance().release();
}

oshmpi_comm* oshmpi_comm::create(std::size_t size,
                                 std::size_t rank,
                                 std::shared_ptr<ccl::kvs_interface> kvs_interface) {
    CCL_THROW_IF_NOT(size > 0 && size <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                     "invalid OSHMPI communicator size: ",
                     size);
    CCL_THROW_IF_NOT(rank < size && rank <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                     "invalid OSHMPI communicator rank: ",
                     rank);

    auto kvs = std::dynamic_pointer_cast<ccl::kvs>(kvs_interface);
    CCL_THROW_IF_NOT(kvs, "only ccl::kvs is allowed with OSHMPI backend");
    ccl::get_kvs_impl_typed<oshmpi_kvs_impl>(kvs);
    return new oshmpi_comm(size, rank, std::move(kvs));
}

ccl::event oshmpi_comm::barrier_impl(const ccl::stream::impl_value_t&,
                                     const ccl::barrier_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    wait_dependencies(deps);
    validate_attributes(attr);
    oshmpi_runtime::instance().barrier();
    return ccl::event{};
}

ccl::event oshmpi_comm::allgather_impl(const void* send_buf,
                                       void* recv_buf,
                                       std::size_t count,
                                       ccl::datatype dtype,
                                       const ccl::stream::impl_value_t&,
                                       const ccl::allgather_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    wait_dependencies(deps);
    validate_attributes(attr);
    const std::size_t bytes = checked_bytes(count, dtype);
    validate_buffer(send_buf, bytes, "allgather send buffer");
    validate_buffer(recv_buf, bytes, "allgather receive buffer");
    oshmpi_runtime::instance().allgather(send_buf, recv_buf, bytes);
    return ccl::event{};
}

ccl::event oshmpi_comm::allgather_impl(const void* send_buf,
                                       const ccl::vector_class<void*>& recv_bufs,
                                       std::size_t count,
                                       ccl::datatype dtype,
                                       const ccl::stream::impl_value_t&,
                                       const ccl::allgather_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    wait_dependencies(deps);
    validate_attributes(attr);
    const std::size_t bytes = checked_bytes(count, dtype);
    validate_buffer(send_buf, bytes, "allgather send buffer");
    for (const auto* buffer : recv_bufs) {
        validate_buffer(buffer, bytes, "allgather receive buffer");
    }
    oshmpi_runtime::instance().allgather(send_buf, recv_bufs, bytes);
    return ccl::event{};
}

ccl::event oshmpi_comm::allreduce_impl(const void* send_buf,
                                       void* recv_buf,
                                       std::size_t count,
                                       ccl::datatype dtype,
                                       ccl::reduction reduction,
                                       const ccl::stream::impl_value_t&,
                                       const ccl::allreduce_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    wait_dependencies(deps);
    validate_attributes(attr);
    const std::size_t bytes = checked_bytes(count, dtype);
    validate_buffer(send_buf, bytes, "allreduce send buffer");
    validate_buffer(recv_buf, bytes, "allreduce receive buffer");
    CCL_THROW_IF_NOT(dtype != ccl::datatype::float16 && dtype != ccl::datatype::bfloat16,
                     "OSHMPI allreduce does not support low-precision datatypes");
    CCL_THROW_IF_NOT(reduction == ccl::reduction::sum || reduction == ccl::reduction::prod ||
                         reduction == ccl::reduction::min || reduction == ccl::reduction::max,
                     "unsupported OSHMPI allreduce reduction: ",
                     static_cast<int>(reduction));
    oshmpi_runtime::instance().allreduce(send_buf, recv_buf, count, dtype, reduction);
    return ccl::event{};
}

ccl::event oshmpi_comm::alltoall_impl(const void* send_buf,
                                      void* recv_buf,
                                      std::size_t count,
                                      ccl::datatype dtype,
                                      const ccl::stream::impl_value_t&,
                                      const ccl::alltoall_attr& attr,
                                      const ccl::vector_class<ccl::event>& deps) {
    wait_dependencies(deps);
    validate_attributes(attr);
    const std::size_t bytes = checked_bytes(count, dtype);
    validate_buffer(send_buf, bytes, "alltoall send buffer");
    validate_buffer(recv_buf, bytes, "alltoall receive buffer");
    oshmpi_runtime::instance().alltoall(send_buf, recv_buf, bytes);
    return ccl::event{};
}

ccl::event oshmpi_comm::broadcast_impl(void* buf,
                                       std::size_t count,
                                       ccl::datatype dtype,
                                       int root,
                                       const ccl::stream::impl_value_t&,
                                       const ccl::broadcast_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    wait_dependencies(deps);
    validate_attributes(attr);
    const std::size_t bytes = checked_bytes(count, dtype);
    validate_buffer(buf, bytes, "broadcast buffer");
    oshmpi_runtime::instance().broadcast(buf, buf, bytes, root);
    return ccl::event{};
}

ccl::event oshmpi_comm::broadcast_impl(void* send_buf,
                                       void* recv_buf,
                                       std::size_t count,
                                       ccl::datatype dtype,
                                       int root,
                                       const ccl::stream::impl_value_t&,
                                       const ccl::broadcast_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    wait_dependencies(deps);
    validate_attributes(attr);
    const std::size_t bytes = checked_bytes(count, dtype);
    if (comm_rank == root) {
        validate_buffer(send_buf, bytes, "broadcast send buffer");
    }
    validate_buffer(recv_buf, bytes, "broadcast receive buffer");
    oshmpi_runtime::instance().broadcast(send_buf, recv_buf, bytes, root);
    return ccl::event{};
}

#define CCL_OSHMPI_UNSUPPORTED_IMPL(name, signature) \
    ccl::event oshmpi_comm::name signature { \
        throw_unsupported(#name); \
    }

CCL_OSHMPI_UNSUPPORTED_IMPL(allgatherv_impl,
                            (const void*,
                             std::size_t,
                             void*,
                             const ccl::vector_class<std::size_t>&,
                             ccl::datatype,
                             const ccl::stream::impl_value_t&,
                             const ccl::allgatherv_attr&,
                             const ccl::vector_class<ccl::event>&))

CCL_OSHMPI_UNSUPPORTED_IMPL(allgatherv_impl,
                            (const void*,
                             std::size_t,
                             const ccl::vector_class<void*>&,
                             const ccl::vector_class<std::size_t>&,
                             ccl::datatype,
                             const ccl::stream::impl_value_t&,
                             const ccl::allgatherv_attr&,
                             const ccl::vector_class<ccl::event>&))

CCL_OSHMPI_UNSUPPORTED_IMPL(alltoall_impl,
                            (const ccl::vector_class<void*>&,
                             const ccl::vector_class<void*>&,
                             std::size_t,
                             ccl::datatype,
                             const ccl::stream::impl_value_t&,
                             const ccl::alltoall_attr&,
                             const ccl::vector_class<ccl::event>&))

CCL_OSHMPI_UNSUPPORTED_IMPL(alltoallv_impl,
                            (const void*,
                             const ccl::vector_class<std::size_t>&,
                             void*,
                             const ccl::vector_class<std::size_t>&,
                             ccl::datatype,
                             const ccl::stream::impl_value_t&,
                             const ccl::alltoallv_attr&,
                             const ccl::vector_class<ccl::event>&))

CCL_OSHMPI_UNSUPPORTED_IMPL(alltoallv_impl,
                            (const ccl::vector_class<void*>&,
                             const ccl::vector_class<std::size_t>&,
                             ccl::vector_class<void*>,
                             const ccl::vector_class<std::size_t>&,
                             ccl::datatype,
                             const ccl::stream::impl_value_t&,
                             const ccl::alltoallv_attr&,
                             const ccl::vector_class<ccl::event>&))

CCL_OSHMPI_UNSUPPORTED_IMPL(reduce_impl,
                            (const void*,
                             void*,
                             std::size_t,
                             ccl::datatype,
                             ccl::reduction,
                             int,
                             const ccl::stream::impl_value_t&,
                             const ccl::reduce_attr&,
                             const ccl::vector_class<ccl::event>&))

CCL_OSHMPI_UNSUPPORTED_IMPL(reduce_scatter_impl,
                            (const void*,
                             void*,
                             std::size_t,
                             ccl::datatype,
                             ccl::reduction,
                             const ccl::stream::impl_value_t&,
                             const ccl::reduce_scatter_attr&,
                             const ccl::vector_class<ccl::event>&))

CCL_OSHMPI_UNSUPPORTED_IMPL(recv_impl,
                            (void*,
                             std::size_t,
                             ccl::datatype,
                             int,
                             const ccl::stream::impl_value_t&,
                             const ccl::pt2pt_attr&,
                             const ccl::vector_class<ccl::event>&))

CCL_OSHMPI_UNSUPPORTED_IMPL(send_impl,
                            (void*,
                             std::size_t,
                             ccl::datatype,
                             int,
                             const ccl::stream::impl_value_t&,
                             const ccl::pt2pt_attr&,
                             const ccl::vector_class<ccl::event>&))

#undef CCL_OSHMPI_UNSUPPORTED_IMPL

} // namespace ccl

#endif // CCL_ENABLE_OSHMPI
