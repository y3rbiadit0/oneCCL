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
#ifdef CCL_ENABLE_NCCL

#include "comm/nccl_comm.hpp"
#include "common/event/impls/event_impl.hpp"
#include "common/event/impls/stub_event.hpp"
#include "nccl_kvs_impl.hpp"
#include "common/log/log.hpp"
#include "oneapi/ccl/api_functions.hpp"

#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(CCL_ENABLE_SYCL)
#include <sycl/sycl.hpp>
#endif

namespace ccl {

namespace {

#if defined(CCL_ENABLE_SYCL)
struct nccl_group_completion {
    std::atomic<bool> ready{ false };
    std::string error;
    sycl::event event;
};

class nccl_group_event_impl final : public ccl::event_impl {
public:
    explicit nccl_group_event_impl(std::shared_ptr<nccl_group_completion> completion)
            : completion(std::move(completion)) {}

    void wait() override {
        CCL_THROW_IF_NOT(completion->ready.load(std::memory_order_acquire),
                         "NCCL group event cannot be waited before ccl::group_end");
        CCL_THROW_IF_NOT(completion->error.empty(), completion->error);
        completion->event.wait();
    }

    bool test() override {
        if (!completion->ready.load(std::memory_order_acquire)) {
            return false;
        }
        CCL_THROW_IF_NOT(completion->error.empty(), completion->error);
        return completion->event.get_info<sycl::info::event::command_execution_status>() ==
               sycl::info::event_command_status::complete;
    }

    bool cancel() override {
        return false;
    }

    ccl::event::native_t& get_native() override {
        CCL_THROW_IF_NOT(completion->ready.load(std::memory_order_acquire),
                         "NCCL group native event is unavailable before ccl::group_end");
        CCL_THROW_IF_NOT(completion->error.empty(), completion->error);
        return completion->event;
    }

private:
    std::shared_ptr<nccl_group_completion> completion;
};

struct nccl_group_stream {
    ccl::stream::impl_value_t stream;
    std::vector<std::shared_ptr<nccl_group_completion>> completions;
};
#endif

struct nccl_group_context {
    bool active = false;
    std::vector<nccl_comm*> communicators;
#if defined(CCL_ENABLE_SYCL)
    std::vector<nccl_group_stream> streams;
#endif
};

thread_local nccl_group_context group_context;

#if defined(CCL_ENABLE_SYCL)
void complete_group(const std::shared_ptr<nccl_group_completion>& completion,
                    sycl::event event,
                    const std::string& error = {}) {
    completion->error = error;
    completion->event = std::move(event);
    completion->ready.store(true, std::memory_order_release);
}
#endif

} // namespace

nccl_comm::nccl_comm(device_t device,
                     context_t context,
                     size_t size,
                     size_t rank,
                     ncclUniqueId nccl_id,
                     std::shared_ptr<ccl::kvs> kvs,
                     const ccl::nccl_kvs_impl* kvs_impl)
        : device_ptr(std::make_shared<ccl::device>(device)),
          context_ptr(std::make_shared<ccl::context>(context)),
          comm_rank(rank),
          comm_size(size),
          kvs(kvs),
          kvs_impl(kvs_impl) {

    LOG_DEBUG("NCCL COMM: initializing communicator for rank ", rank, "/", size);

    ncclResult_t status = ncclCommInitRank(&nccl_comm_handle, size, nccl_id, rank);
    CCL_THROW_IF_NOT(status == ncclSuccess,
                     "ncclCommInitRank failed: ", ncclGetErrorString(status));

    LOG_INFO("NCCL COMM: communicator initialized successfully for rank ", rank, "/", size);
}

nccl_comm::~nccl_comm() {
    LOG_DEBUG("NCCL COMM: destroying communicator for rank ", comm_rank);

    if (nccl_comm_handle != nullptr) {
        ncclResult_t status = ncclCommDestroy(nccl_comm_handle);
        if (status != ncclSuccess) {
            LOG_WARN("NCCL COMM: ncclCommDestroy failed: ", ncclGetErrorString(status));
        }
    }
}

nccl_comm* nccl_comm::create(device_t device,
                             context_t context,
                             size_t size,
                             size_t rank,
                             std::shared_ptr<ccl::kvs_interface> kvs_interface) {
    auto kvs_inst = std::dynamic_pointer_cast<ccl::kvs>(kvs_interface);
    CCL_THROW_IF_NOT(kvs_inst != nullptr, "only ccl::kvs is allowed with NCCL backend");

    auto kvs_impl = ccl::get_kvs_impl_typed<nccl_kvs_impl>(kvs_inst);

    ncclUniqueId nccl_id = kvs_impl->get_nccl_id();

    return new nccl_comm(device, context, size, rank, nccl_id, std::move(kvs_inst), kvs_impl);
}

/* create a ccl::event that tracks completion of all work submitted to the SYCL queue */
ccl::event nccl_comm::make_event(const ccl::stream::impl_value_t& stream) {
#if defined(CCL_ENABLE_SYCL)
    auto sycl_queue = stream->get_native_stream();
    sycl::event sycl_ev = sycl_queue.ext_oneapi_submit_barrier();
    LOG_DEBUG("NCCL COMM: created SYCL barrier event for stream synchronization");
    return ccl::event::create_from_native(sycl_ev);
#else
    LOG_DEBUG("NCCL COMM: SYCL not enabled, returning stub event");
    return std::unique_ptr<ccl::event_impl>(new ccl::stub_event_impl());
#endif
}

void nccl_comm::group_start() {
#if !defined(CCL_ENABLE_SYCL)
    CCL_THROW("NCCL groups require SYCL support");
#else
    CCL_THROW_IF_NOT(!group_context.active, "nested NCCL groups are not supported");

    group_context.communicators.clear();
    group_context.streams.clear();

    ncclResult_t status = ncclGroupStart();
    CCL_THROW_IF_NOT(status == ncclSuccess, "ncclGroupStart failed: ", ncclGetErrorString(status));

    group_context.active = true;
#endif
}

void nccl_comm::group_end() {
#if !defined(CCL_ENABLE_SYCL)
    CCL_THROW("NCCL groups require SYCL support");
#else
    CCL_THROW_IF_NOT(group_context.active, "NCCL group_end called without group_start");

    auto communicators = std::move(group_context.communicators);
    auto streams = std::move(group_context.streams);

    ncclResult_t status = ncclGroupEnd();

    group_context.active = false;
    group_context.communicators.clear();
    group_context.streams.clear();

    if (status != ncclSuccess) {
        const std::string error = std::string("ncclGroupEnd failed: ") + ncclGetErrorString(status);
        for (auto* comm : communicators) {
            if (comm->nccl_comm_handle != nullptr) {
                const ncclResult_t abort_status = ncclCommAbort(comm->nccl_comm_handle);
                if (abort_status != ncclSuccess) {
                    LOG_WARN("NCCL COMM: ncclCommAbort failed: ", ncclGetErrorString(abort_status));
                }
                comm->nccl_comm_handle = nullptr;
            }
        }
        for (auto& stream : streams) {
            for (auto& completion : stream.completions) {
                complete_group(completion, sycl::event{}, error);
            }
        }
        CCL_THROW(error);
    }

    try {
        std::vector<sycl::event> events;
        events.reserve(streams.size());
        for (auto& stream : streams) {
            events.push_back(stream.stream->get_native_stream().ext_oneapi_submit_barrier());
        }
        for (std::size_t index = 0; index < streams.size(); ++index) {
            for (auto& completion : streams[index].completions) {
                complete_group(completion, events[index]);
            }
        }
    }
    catch (const std::exception& error) {
        const std::string message =
            std::string("NCCL group stream synchronization failed: ") + error.what();
        for (auto& stream : streams) {
            for (auto& completion : stream.completions) {
                complete_group(completion, sycl::event{}, message);
            }
        }
        throw;
    }
#endif
}

bool nccl_comm::is_group_active() {
    return group_context.active;
}

void nccl_comm::register_group_operation(const ccl::stream::impl_value_t& stream) {
    CCL_THROW_IF_NOT(group_context.active, "NCCL group stream registered outside a group");

    const auto communicator =
        std::find(group_context.communicators.begin(), group_context.communicators.end(), this);
    if (communicator == group_context.communicators.end()) {
        group_context.communicators.push_back(this);
    }

#if defined(CCL_ENABLE_SYCL)
    const auto existing = std::find_if(
        group_context.streams.begin(), group_context.streams.end(), [&](const auto& item) {
            return item.stream.get() == stream.get();
        });
    if (existing == group_context.streams.end()) {
        group_context.streams.push_back({ stream, {} });
    }
#else
    CCL_THROW("NCCL groups require SYCL support");
#endif
}

ccl::event nccl_comm::make_group_event(const ccl::stream::impl_value_t& stream) {
    CCL_THROW_IF_NOT(group_context.active, "NCCL group event created outside a group");
#if defined(CCL_ENABLE_SYCL)
    const auto item = std::find_if(
        group_context.streams.begin(), group_context.streams.end(), [&](const auto& value) {
            return value.stream.get() == stream.get();
        });
    CCL_THROW_IF_NOT(item != group_context.streams.end(),
                     "NCCL group event stream was not registered");

    auto completion = std::make_shared<nccl_group_completion>();
    item->completions.push_back(completion);
    ccl::event::impl_value_t implementation =
        std::make_unique<nccl_group_event_impl>(std::move(completion));
    return ccl::event{ std::move(implementation) };
#else
    CCL_THROW("NCCL groups require SYCL support");
#endif
}

ccl::event nccl_comm::make_operation_event(const ccl::stream::impl_value_t& stream) {
    if (!is_group_active()) {
        return make_event(stream);
    }

    register_group_operation(stream);
    return make_group_event(stream);
}

/* barrier */
ccl::event nccl_comm::barrier_impl(const ccl::stream::impl_value_t& stream,
                                   const ccl::barrier_attr& attr,
                                   const ccl::vector_class<ccl::event>& deps) {
    CCL_THROW_IF_NOT(!is_group_active(), "NCCL barrier is not supported inside a group");
    // NCCL does not have a native barrier, simulate with allreduce on a single element
    LOG_DEBUG("NCCL COMM: barrier (simulated with allreduce)");

    cudaStream_t cuda_stream = get_cuda_stream(stream);

    // Allocate device memory via SYCL to avoid direct cudart link dependency
    auto sycl_queue = stream->get_native_stream();
    int* d_dummy = sycl::malloc_device<int>(1, sycl_queue);
    CCL_THROW_IF_NOT(d_dummy != nullptr,
                     "sycl::malloc_device for barrier dummy buffer failed");

    sycl_queue.memset(d_dummy, 0, sizeof(int)).wait();

    ncclResult_t status = ncclAllReduce(d_dummy, d_dummy, 1, ncclInt, ncclSum,
                                        nccl_comm_handle, cuda_stream);
    CCL_THROW_IF_NOT(status == ncclSuccess,
                     "NCCL barrier (allreduce) failed: ", ncclGetErrorString(status));

    // Submit barrier to track completion, then free the temporary buffer after sync
    auto ev = make_event(stream);

    // We need to synchronize before freeing the dummy buffer
    sycl_queue.ext_oneapi_submit_barrier().wait();
    sycl::free(d_dummy, sycl_queue);

    return ev;
}

/* allreduce */
ccl::event nccl_comm::allreduce_impl(const void* send_buf,
                                     void* recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     ccl::reduction reduction,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allreduce_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("NCCL COMM: allreduce count=", count, " dtype=", static_cast<int>(dtype),
              " reduction=", static_cast<int>(reduction));

    ncclDataType_t nccl_dtype = get_nccl_datatype(dtype);
    ncclRedOp_t nccl_op = get_nccl_reduction(reduction);
    cudaStream_t cuda_stream = get_cuda_stream(stream);

    ncclResult_t status = ncclAllReduce(send_buf, recv_buf, count, nccl_dtype, nccl_op,
                                        nccl_comm_handle, cuda_stream);
    CCL_THROW_IF_NOT(status == ncclSuccess,
                     "ncclAllReduce failed: ", ncclGetErrorString(status));

    LOG_DEBUG("NCCL COMM: allreduce submitted successfully");

    return make_operation_event(stream);
}

/* extract cudaStream_t from SYCL queue */
cudaStream_t nccl_comm::get_cuda_stream(const ccl::stream::impl_value_t& stream) {
#if defined(CCL_ENABLE_SYCL)
    try {
        auto sycl_queue = stream->get_native_stream();
        CCL_THROW_IF_NOT(sycl_queue.has_property<sycl::property::queue::in_order>(),
                         "NCCL backend requires an in-order SYCL queue");
        auto cuda_stream = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue);

        LOG_DEBUG("NCCL COMM: extracted CUDA stream: ", cuda_stream);
        return cuda_stream;
    }
    catch (const std::exception& e) {
        CCL_THROW("Failed to extract CUDA stream from SYCL queue: ", e.what());
    }
#else
    CCL_THROW("SYCL support not enabled, cannot extract CUDA stream");
#endif
}

/* convert ccl::datatype -> ncclDataType_t */
ncclDataType_t nccl_comm::get_nccl_datatype(ccl::datatype dtype) {
    switch (dtype) {
        case ccl::datatype::int8:
            return ncclInt8;
        case ccl::datatype::uint8:
            return ncclUint8;
        case ccl::datatype::int32:
            return ncclInt32;
        case ccl::datatype::uint32:
            return ncclUint32;
        case ccl::datatype::int64:
            return ncclInt64;
        case ccl::datatype::uint64:
            return ncclUint64;
        case ccl::datatype::float16:
            return ncclFloat16;
        case ccl::datatype::float32:
            return ncclFloat32;
        case ccl::datatype::float64:
            return ncclFloat64;
        case ccl::datatype::bfloat16:
            return ncclBfloat16;
        default:
            CCL_THROW("Unsupported datatype for NCCL: ", static_cast<int>(dtype));
    }
}

/* convert ccl::reduction -> ncclRedOp_t */
ncclRedOp_t nccl_comm::get_nccl_reduction(ccl::reduction reduction) {
    switch (reduction) {
        case ccl::reduction::sum:
            return ncclSum;
        case ccl::reduction::prod:
            return ncclProd;
        case ccl::reduction::min:
            return ncclMin;
        case ccl::reduction::max:
            return ncclMax;
        default:
            CCL_THROW("Unsupported reduction operation for NCCL: ", static_cast<int>(reduction));
    }
}

// stub implementations for collectives not yet supported
#define NCCL_COMM_STUB_IMPL(name) \
    CCL_THROW(#name " is not implemented for NCCL backend yet");

/* allgather */
ccl::event nccl_comm::allgather_impl(const void* send_buf,
                                     void* recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allgather_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    NCCL_COMM_STUB_IMPL(allgather);
}

ccl::event nccl_comm::allgather_impl(const void* send_buf,
                                     const ccl::vector_class<void*>& recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allgather_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    NCCL_COMM_STUB_IMPL(allgather);
}

/* allgatherv */
ccl::event nccl_comm::allgatherv_impl(const void* send_buf,
                                      size_t send_count,
                                      void* recv_buf,
                                      const ccl::vector_class<size_t>& recv_counts,
                                      ccl::datatype dtype,
                                      const ccl::stream::impl_value_t& stream,
                                      const ccl::allgatherv_attr& attr,
                                      const ccl::vector_class<ccl::event>& deps) {
    NCCL_COMM_STUB_IMPL(allgatherv);
}

ccl::event nccl_comm::allgatherv_impl(const void* send_buf,
                                      size_t send_count,
                                      const ccl::vector_class<void*>& recv_bufs,
                                      const ccl::vector_class<size_t>& recv_counts,
                                      ccl::datatype dtype,
                                      const ccl::stream::impl_value_t& stream,
                                      const ccl::allgatherv_attr& attr,
                                      const ccl::vector_class<ccl::event>& deps) {
    NCCL_COMM_STUB_IMPL(allgatherv);
}

/* alltoall */
ccl::event nccl_comm::alltoall_impl(const void* send_buf,
                                    void* recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::alltoall_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("NCCL COMM: alltoall count=", count, " dtype=", static_cast<int>(dtype));

    ncclDataType_t nccl_dtype = get_nccl_datatype(dtype);
    cudaStream_t cuda_stream = get_cuda_stream(stream);

#if CCL_NCCL_ALLTOALL_SUPPORTED
    int version = 0;
    ncclResult_t version_status = ncclGetVersion(&version);
    if (version_status == ncclSuccess && version >= NCCL_VERSION(2, 28, 0)) {
        auto alltoall_fn = ncclGetAllToAll();
        if (alltoall_fn) {
            LOG_DEBUG("NCCL COMM: using ncclAlltoAll native implementation");
            ncclResult_t status =
                alltoall_fn(send_buf, recv_buf, count, nccl_dtype, nccl_comm_handle, cuda_stream);
            CCL_THROW_IF_NOT(status == ncclSuccess,
                             "ncclAlltoAll failed: ", ncclGetErrorString(status));
            return make_operation_event(stream);
        }
        else {
            LOG_DEBUG("NCCL COMM: ncclAlltoAll symbol not found, fallback to send/recv");
        }
    }
    else if (version_status != ncclSuccess) {
        LOG_DEBUG("NCCL COMM: ncclGetVersion failed, fallback to send/recv: ",
                  ncclGetErrorString(version_status));
    }
#endif
    // fallback: implement alltoall with send/recv inside a group
    LOG_DEBUG("NCCL COMM: using send/recv fallback implementation for alltoall");
    const size_t dtype_size = ccl::get_datatype_size(dtype);
    const size_t block_size = count * dtype_size;
    const char* send_base = static_cast<const char*>(send_buf);
    char* recv_base = static_cast<char*>(recv_buf);

    const bool outer_group = is_group_active();
    ncclResult_t status = ncclSuccess;
    if (!outer_group) {
        status = ncclGroupStart();
        CCL_THROW_IF_NOT(
            status == ncclSuccess, "ncclGroupStart failed: ", ncclGetErrorString(status));
    }

    ncclResult_t first_error = ncclSuccess;
    for (size_t peer = 0; peer < comm_size; ++peer) {
        const void* send_ptr = send_base + (peer * block_size);
        void* recv_ptr = recv_base + (peer * block_size);

        status = ncclSend(send_ptr, count, nccl_dtype, static_cast<int>(peer), nccl_comm_handle,
                          cuda_stream);
        if (status != ncclSuccess && first_error == ncclSuccess) {
            first_error = status;
        }

        status = ncclRecv(recv_ptr, count, nccl_dtype, static_cast<int>(peer), nccl_comm_handle,
                          cuda_stream);
        if (status != ncclSuccess && first_error == ncclSuccess) {
            first_error = status;
        }
    }

    if (!outer_group) {
        status = ncclGroupEnd();
        if (first_error == ncclSuccess) {
            first_error = status;
        }
    }

    CCL_THROW_IF_NOT(first_error == ncclSuccess,
                     "ncclAllToAll (send/recv) failed: ", ncclGetErrorString(first_error));

    return make_operation_event(stream);
}

/* alltoallv */
ccl::event nccl_comm::alltoallv_impl(const void* send_buf,
                                     const ccl::vector_class<size_t>& send_counts,
                                     void* recv_buf,
                                     const ccl::vector_class<size_t>& recv_counts,
                                     ccl::datatype dtype,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::alltoallv_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    NCCL_COMM_STUB_IMPL(alltoallv);
}

/* broadcast */
ccl::event nccl_comm::broadcast_impl(void* buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     int root,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::broadcast_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    NCCL_COMM_STUB_IMPL(broadcast);
}

ccl::event nccl_comm::broadcast_impl(void* send_buf,
                                     void* recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     int root,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::broadcast_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    NCCL_COMM_STUB_IMPL(broadcast);
}

/* reduce */
ccl::event nccl_comm::reduce_impl(const void* send_buf,
                                  void* recv_buf,
                                  size_t count,
                                  ccl::datatype dtype,
                                  ccl::reduction reduction,
                                  int root,
                                  const ccl::stream::impl_value_t& stream,
                                  const ccl::reduce_attr& attr,
                                  const ccl::vector_class<ccl::event>& deps) {
    NCCL_COMM_STUB_IMPL(reduce);
}

/* reduce_scatter */
ccl::event nccl_comm::reduce_scatter_impl(const void* send_buf,
                                          void* recv_buf,
                                          size_t recv_count,
                                          ccl::datatype dtype,
                                          ccl::reduction reduction,
                                          const ccl::stream::impl_value_t& stream,
                                          const ccl::reduce_scatter_attr& attr,
                                          const ccl::vector_class<ccl::event>& deps) {
    NCCL_COMM_STUB_IMPL(reduce_scatter);
}

/* recv */
ccl::event nccl_comm::recv_impl(void* recv_buf,
                                size_t recv_count,
                                ccl::datatype dtype,
                                int peer,
                                const ccl::stream::impl_value_t& stream,
                                const ccl::pt2pt_attr& attr,
                                const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("NCCL COMM: recv count=", recv_count, " dtype=", static_cast<int>(dtype),
              " peer=", peer);

    ncclDataType_t nccl_dtype = get_nccl_datatype(dtype);
    cudaStream_t cuda_stream = get_cuda_stream(stream);

    ncclResult_t status =
        ncclRecv(recv_buf, recv_count, nccl_dtype, peer, nccl_comm_handle, cuda_stream);
    CCL_THROW_IF_NOT(status == ncclSuccess,
                     "ncclRecv failed: ", ncclGetErrorString(status));

    return make_operation_event(stream);
}

/* send */
ccl::event nccl_comm::send_impl(void* send_buf,
                                size_t send_count,
                                ccl::datatype dtype,
                                int peer,
                                const ccl::stream::impl_value_t& stream,
                                const ccl::pt2pt_attr& attr,
                                const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("NCCL COMM: send count=", send_count, " dtype=", static_cast<int>(dtype),
              " peer=", peer);

    ncclDataType_t nccl_dtype = get_nccl_datatype(dtype);
    cudaStream_t cuda_stream = get_cuda_stream(stream);

    ncclResult_t status =
        ncclSend(send_buf, send_count, nccl_dtype, peer, nccl_comm_handle, cuda_stream);
    CCL_THROW_IF_NOT(status == ncclSuccess,
                     "ncclSend failed: ", ncclGetErrorString(status));

    return make_operation_event(stream);
}

} // namespace ccl

#endif // CCL_ENABLE_NCCL
