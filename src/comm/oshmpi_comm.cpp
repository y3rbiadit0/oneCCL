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
#include "common/oshmpi/oshmpi_device.hpp"
#include "common/oshmpi/oshmpi_runtime.hpp"
// ccl::stream::impl_value_t is a shared_ptr to ccl_stream, which the public
// headers only forward declare. wait_stream() calls through it, so the full
// definition is required here.
#include "common/stream/stream.hpp"
#include "oshmpi_kvs_impl.hpp"

namespace ccl {
namespace {

void wait_dependencies(const ccl::vector_class<ccl::event>& deps) {
    for (const auto& dependency : deps) {
        const_cast<ccl::event&>(dependency).wait();
    }
}

/* The collectives are blocking and stage through host memory, so anything the
 * caller queued on its own stream is not otherwise ordered against them. Draining
 * the queue first is what makes a device operand safe to read. This is a
 * correctness requirement, not a tuning choice: without it the backend can stage a
 * buffer the caller's kernel has not finished writing. */
void wait_stream(const ccl::stream::impl_value_t& stream) {
#ifdef CCL_ENABLE_SYCL
    if (stream && stream->is_sycl_device_stream()) {
        stream->get_native_stream().wait();
    }
#else // CCL_ENABLE_SYCL
    (void)stream;
#endif // CCL_ENABLE_SYCL
}

/* Every collective begins the same way: settle the caller's dependencies, drain
 * its stream, then check the attributes it asked for are ones we honour. */
template <class attr_type>
void enter_collective(const ccl::stream::impl_value_t& stream,
                      const attr_type& attr,
                      const ccl::vector_class<ccl::event>& deps);

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

template <class attr_type>
void enter_collective(const ccl::stream::impl_value_t& stream,
                      const attr_type& attr,
                      const ccl::vector_class<ccl::event>& deps) {
    wait_dependencies(deps);
    wait_stream(stream);
    validate_attributes(attr);
}

/* Builds what the runtime needs in order to touch device memory. The queue comes
 * from the caller's stream and the context from that same queue, so classification
 * and copying always agree about which context owns an allocation.
 *
 * A device communicator used without a stream still yields a context, which is
 * deliberate: the runtime can then recognise a device operand and report it as an
 * error instead of memcpy'ing device memory on the host. A host communicator
 * yields the default accessor, under which every buffer is host memory. */
oshmpi_device::accessor make_accessor(const ccl::stream::impl_value_t& stream,
                                      const std::shared_ptr<ccl::context>& context) {
#ifdef CCL_ENABLE_SYCL
    if (stream && stream->is_sycl_device_stream()) {
        // get_native_stream() returns by value; sycl::queue is a reference-counted
        // handle, so the copy still refers to the caller's queue.
        sycl::queue queue = stream->get_native_stream();
        return oshmpi_device::accessor(queue.get_context(), queue);
    }
    if (context) {
        return oshmpi_device::accessor(context->get_native());
    }
#else // CCL_ENABLE_SYCL
    (void)stream;
    (void)context;
#endif // CCL_ENABLE_SYCL
    return oshmpi_device::accessor{};
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

oshmpi_comm::oshmpi_comm(std::size_t size,
                         std::size_t rank,
                         std::shared_ptr<ccl::kvs> kvs,
                         device_ptr_t device,
                         context_ptr_t context)
        : comm_rank(static_cast<int>(rank)),
          comm_size(static_cast<int>(size)),
          kvs(std::move(kvs)),
          device_ptr(std::move(device)),
          context_ptr(std::move(context)) {
    oshmpi_runtime::instance().acquire(size, rank);
}

oshmpi_comm::~oshmpi_comm() {
    oshmpi_runtime::instance().release();
}

namespace {

std::shared_ptr<ccl::kvs> validate_and_take_kvs(std::size_t size,
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
    return kvs;
}

} // namespace

oshmpi_comm* oshmpi_comm::create(device_t device,
                                 context_t context,
                                 std::size_t size,
                                 std::size_t rank,
                                 std::shared_ptr<ccl::kvs_interface> kvs_interface) {
    auto kvs = validate_and_take_kvs(size, rank, std::move(kvs_interface));
    return new oshmpi_comm(size,
                           rank,
                           std::move(kvs),
                           std::make_shared<ccl::device>(device),
                           std::make_shared<ccl::context>(context));
}

oshmpi_comm* oshmpi_comm::create(std::size_t size,
                                 std::size_t rank,
                                 std::shared_ptr<ccl::kvs_interface> kvs_interface) {
    auto kvs = validate_and_take_kvs(size, rank, std::move(kvs_interface));
    return new oshmpi_comm(size, rank, std::move(kvs));
}

ccl::event oshmpi_comm::barrier_impl(const ccl::stream::impl_value_t& stream,
                                     const ccl::barrier_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    enter_collective(stream, attr, deps);
    oshmpi_runtime::instance().barrier();
    return ccl::event{};
}

ccl::event oshmpi_comm::allgather_impl(const void* send_buf,
                                       void* recv_buf,
                                       std::size_t count,
                                       ccl::datatype dtype,
                                       const ccl::stream::impl_value_t& stream,
                                       const ccl::allgather_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    enter_collective(stream, attr, deps);
    const std::size_t bytes = checked_bytes(count, dtype);
    validate_buffer(send_buf, bytes, "allgather send buffer");
    validate_buffer(recv_buf, bytes, "allgather receive buffer");
    oshmpi_runtime::instance().allgather(send_buf, recv_buf, bytes, make_accessor(stream, context_ptr));
    return ccl::event{};
}

ccl::event oshmpi_comm::allgather_impl(const void* send_buf,
                                       const ccl::vector_class<void*>& recv_bufs,
                                       std::size_t count,
                                       ccl::datatype dtype,
                                       const ccl::stream::impl_value_t& stream,
                                       const ccl::allgather_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    enter_collective(stream, attr, deps);
    const std::size_t bytes = checked_bytes(count, dtype);
    validate_buffer(send_buf, bytes, "allgather send buffer");
    for (const auto* buffer : recv_bufs) {
        validate_buffer(buffer, bytes, "allgather receive buffer");
    }
    oshmpi_runtime::instance().allgather(send_buf, recv_bufs, bytes, make_accessor(stream, context_ptr));
    return ccl::event{};
}

ccl::event oshmpi_comm::allreduce_impl(const void* send_buf,
                                       void* recv_buf,
                                       std::size_t count,
                                       ccl::datatype dtype,
                                       ccl::reduction reduction,
                                       const ccl::stream::impl_value_t& stream,
                                       const ccl::allreduce_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    enter_collective(stream, attr, deps);
    const std::size_t bytes = checked_bytes(count, dtype);
    validate_buffer(send_buf, bytes, "allreduce send buffer");
    validate_buffer(recv_buf, bytes, "allreduce receive buffer");
    CCL_THROW_IF_NOT(dtype != ccl::datatype::float16 && dtype != ccl::datatype::bfloat16,
                     "OSHMPI allreduce does not support low-precision datatypes");
    CCL_THROW_IF_NOT(reduction == ccl::reduction::sum || reduction == ccl::reduction::prod ||
                         reduction == ccl::reduction::min || reduction == ccl::reduction::max,
                     "unsupported OSHMPI allreduce reduction: ",
                     static_cast<int>(reduction));
    oshmpi_runtime::instance().allreduce(
        send_buf, recv_buf, count, dtype, reduction, make_accessor(stream, context_ptr));
    return ccl::event{};
}

ccl::event oshmpi_comm::alltoall_impl(const void* send_buf,
                                      void* recv_buf,
                                      std::size_t count,
                                      ccl::datatype dtype,
                                      const ccl::stream::impl_value_t& stream,
                                      const ccl::alltoall_attr& attr,
                                      const ccl::vector_class<ccl::event>& deps) {
    enter_collective(stream, attr, deps);
    const std::size_t bytes = checked_bytes(count, dtype);
    validate_buffer(send_buf, bytes, "alltoall send buffer");
    validate_buffer(recv_buf, bytes, "alltoall receive buffer");
    oshmpi_runtime::instance().alltoall(send_buf, recv_buf, bytes, make_accessor(stream, context_ptr));
    return ccl::event{};
}

ccl::event oshmpi_comm::broadcast_impl(void* buf,
                                       std::size_t count,
                                       ccl::datatype dtype,
                                       int root,
                                       const ccl::stream::impl_value_t& stream,
                                       const ccl::broadcast_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    enter_collective(stream, attr, deps);
    const std::size_t bytes = checked_bytes(count, dtype);
    validate_buffer(buf, bytes, "broadcast buffer");
    oshmpi_runtime::instance().broadcast(buf, buf, bytes, root, make_accessor(stream, context_ptr));
    return ccl::event{};
}

ccl::event oshmpi_comm::broadcast_impl(void* send_buf,
                                       void* recv_buf,
                                       std::size_t count,
                                       ccl::datatype dtype,
                                       int root,
                                       const ccl::stream::impl_value_t& stream,
                                       const ccl::broadcast_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    enter_collective(stream, attr, deps);
    const std::size_t bytes = checked_bytes(count, dtype);
    if (comm_rank == root) {
        validate_buffer(send_buf, bytes, "broadcast send buffer");
    }
    validate_buffer(recv_buf, bytes, "broadcast receive buffer");
    oshmpi_runtime::instance().broadcast(
        send_buf, recv_buf, bytes, root, make_accessor(stream, context_ptr));
    return ccl::event{};
}

ccl::event oshmpi_comm::send_impl(void* send_buf,
                                  std::size_t send_count,
                                  ccl::datatype dtype,
                                  int peer,
                                  const ccl::stream::impl_value_t& stream,
                                  const ccl::pt2pt_attr& attr,
                                  const ccl::vector_class<ccl::event>& deps) {
    enter_collective(stream, attr, deps);
    const std::size_t bytes = checked_bytes(send_count, dtype);
    validate_buffer(send_buf, bytes, "send buffer");
    CCL_THROW_IF_NOT(peer >= 0 && peer < comm_size, "invalid send peer: ", peer);
    oshmpi_runtime::instance().send(send_buf, bytes, peer, make_accessor(stream, context_ptr));
    return ccl::event{};
}

ccl::event oshmpi_comm::recv_impl(void* recv_buf,
                                  std::size_t recv_count,
                                  ccl::datatype dtype,
                                  int peer,
                                  const ccl::stream::impl_value_t& stream,
                                  const ccl::pt2pt_attr& attr,
                                  const ccl::vector_class<ccl::event>& deps) {
    enter_collective(stream, attr, deps);
    const std::size_t bytes = checked_bytes(recv_count, dtype);
    validate_buffer(recv_buf, bytes, "receive buffer");
    CCL_THROW_IF_NOT(peer >= 0 && peer < comm_size, "invalid recv peer: ", peer);
    oshmpi_runtime::instance().recv(recv_buf, bytes, peer, make_accessor(stream, context_ptr));
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
                             const ccl::stream::impl_value_t& stream,
                             const ccl::allgatherv_attr&,
                             const ccl::vector_class<ccl::event>&))

CCL_OSHMPI_UNSUPPORTED_IMPL(allgatherv_impl,
                            (const void*,
                             std::size_t,
                             const ccl::vector_class<void*>&,
                             const ccl::vector_class<std::size_t>&,
                             ccl::datatype,
                             const ccl::stream::impl_value_t& stream,
                             const ccl::allgatherv_attr&,
                             const ccl::vector_class<ccl::event>&))

CCL_OSHMPI_UNSUPPORTED_IMPL(alltoall_impl,
                            (const ccl::vector_class<void*>&,
                             const ccl::vector_class<void*>&,
                             std::size_t,
                             ccl::datatype,
                             const ccl::stream::impl_value_t& stream,
                             const ccl::alltoall_attr&,
                             const ccl::vector_class<ccl::event>&))

CCL_OSHMPI_UNSUPPORTED_IMPL(alltoallv_impl,
                            (const void*,
                             const ccl::vector_class<std::size_t>&,
                             void*,
                             const ccl::vector_class<std::size_t>&,
                             ccl::datatype,
                             const ccl::stream::impl_value_t& stream,
                             const ccl::alltoallv_attr&,
                             const ccl::vector_class<ccl::event>&))

CCL_OSHMPI_UNSUPPORTED_IMPL(alltoallv_impl,
                            (const ccl::vector_class<void*>&,
                             const ccl::vector_class<std::size_t>&,
                             ccl::vector_class<void*>,
                             const ccl::vector_class<std::size_t>&,
                             ccl::datatype,
                             const ccl::stream::impl_value_t& stream,
                             const ccl::alltoallv_attr&,
                             const ccl::vector_class<ccl::event>&))

CCL_OSHMPI_UNSUPPORTED_IMPL(reduce_impl,
                            (const void*,
                             void*,
                             std::size_t,
                             ccl::datatype,
                             ccl::reduction,
                             int,
                             const ccl::stream::impl_value_t& stream,
                             const ccl::reduce_attr&,
                             const ccl::vector_class<ccl::event>&))

CCL_OSHMPI_UNSUPPORTED_IMPL(reduce_scatter_impl,
                            (const void*,
                             void*,
                             std::size_t,
                             ccl::datatype,
                             ccl::reduction,
                             const ccl::stream::impl_value_t& stream,
                             const ccl::reduce_scatter_attr&,
                             const ccl::vector_class<ccl::event>&))

#undef CCL_OSHMPI_UNSUPPORTED_IMPL

} // namespace ccl

#endif // CCL_ENABLE_OSHMPI
