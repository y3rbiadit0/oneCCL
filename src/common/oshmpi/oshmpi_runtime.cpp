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

#include "common/oshmpi/oshmpi_runtime.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#include <shmem.h>

// api_functions.hpp declares get_datatype_size but includes nothing itself - it
// expects the umbrella header to have established the type stack first.
#include "oneapi/ccl.hpp"

#include "common/datatype/datatype.hpp"
#include "common/log/log.hpp"

namespace ccl {
namespace {

constexpr std::size_t default_staging_size = 64UL * 1024UL * 1024UL;
constexpr std::size_t staging_alignment = 64;

/* Slots in the symmetric scratch block used to agree on startup parameters.
 * OpenSHMEM only guarantees remote accessibility for the symmetric heap and the
 * executable's data segment - a shared library's statics are not symmetric, so
 * these reduction operands must come from shmem_malloc. */
enum scratch_slot {
    scratch_preflight_source,
    scratch_preflight_result,
    scratch_staging_request,
    scratch_staging_min,
    scratch_staging_max,
    scratch_slot_count
};

std::size_t checked_multiply(std::size_t left, std::size_t right, const char* description) {
    CCL_THROW_IF_NOT(right == 0 || left <= std::numeric_limits<std::size_t>::max() / right,
                     description,
                     " size overflow");
    return left * right;
}

std::size_t parse_staging_size() {
    const char* value = std::getenv("CCL_OSHMPI_STAGING_SIZE");
    if (!value || !*value) {
        return default_staging_size;
    }

    CCL_THROW_IF_NOT(*value != '-', "invalid CCL_OSHMPI_STAGING_SIZE: ", value);
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    CCL_THROW_IF_NOT(errno == 0 && end != value, "invalid CCL_OSHMPI_STAGING_SIZE: ", value);

    unsigned long long multiplier = 1;
    if (*end != '\0') {
        CCL_THROW_IF_NOT(end[1] == '\0', "invalid CCL_OSHMPI_STAGING_SIZE suffix: ", value);
        switch (*end) {
            case 'k':
            case 'K':
                multiplier = 1024ULL;
                break;
            case 'm':
            case 'M':
                multiplier = 1024ULL * 1024ULL;
                break;
            case 'g':
            case 'G':
                multiplier = 1024ULL * 1024ULL * 1024ULL;
                break;
            default:
                CCL_THROW("invalid CCL_OSHMPI_STAGING_SIZE suffix: ", value);
        }
    }

    CCL_THROW_IF_NOT(parsed > 0 &&
                         parsed <= std::numeric_limits<std::size_t>::max() / multiplier,
                     "CCL_OSHMPI_STAGING_SIZE is too large: ",
                     value);
    return static_cast<std::size_t>(parsed * multiplier);
}

void check_status(int status, const char* operation) {
    CCL_THROW_IF_NOT(status == SHMEM_SUCCESS, operation, " failed with status ", status);
}

} // namespace

oshmpi_runtime& oshmpi_runtime::instance() {
    static oshmpi_runtime runtime;
    return runtime;
}

void oshmpi_runtime::acquire(std::size_t size, std::size_t rank) {
    std::lock_guard<std::mutex> lock(state_mutex);

    CCL_THROW_IF_NOT(!finalized,
                     "OSHMPI backend cannot be reinitialized after final communicator teardown");

    if (users > 0) {
        CCL_THROW_IF_NOT(size == static_cast<std::size_t>(world_size) &&
                             rank == static_cast<std::size_t>(world_rank),
                         "all OSHMPI communicators must represent SHMEM_TEAM_WORLD");
        ++users;
        return;
    }

    int provided = SHMEM_THREAD_SINGLE;
    const int status = shmem_init_thread(SHMEM_THREAD_SERIALIZED, &provided);
    // A failed init leaves no runtime to run collectives on, so there is nothing
    // to coordinate with the other PEs - report it directly.
    CCL_THROW_IF_NOT(status == SHMEM_SUCCESS, "shmem_init_thread failed with status ", status);
    initialized = true;

    try {
        world_rank = shmem_my_pe();
        world_size = shmem_n_pes();

        // Collective, and must run on every PE before any branch that can throw.
        scratch = static_cast<std::uint64_t*>(
            shmem_malloc(sizeof(std::uint64_t) * scratch_slot_count));
        CCL_THROW_IF_NOT(scratch,
                         "shmem_malloc failed for OSHMPI scratch; increase SHMEM_SYMMETRIC_SIZE");

        bool local_preflight = provided >= SHMEM_THREAD_SERIALIZED && world_rank >= 0 &&
                               world_size > 0 && size == static_cast<std::size_t>(world_size) &&
                               rank == static_cast<std::size_t>(world_rank);

        std::size_t requested_size = 0;
        try {
            requested_size = parse_staging_size();
        }
        catch (const ccl::exception&) {
            local_preflight = false;
        }

        scratch[scratch_preflight_source] = local_preflight ? 1 : 0;
        check_status(shmem_uint64_min_reduce(SHMEM_TEAM_WORLD,
                                             &scratch[scratch_preflight_result],
                                             &scratch[scratch_preflight_source],
                                             1),
                     "OSHMPI initialization preflight");
        CCL_THROW_IF_NOT(scratch[scratch_preflight_result] == 1,
                         "OSHMPI initialization parameters are inconsistent or invalid; "
                         "check rank, size, thread level, and CCL_OSHMPI_STAGING_SIZE on every PE");

        scratch[scratch_staging_request] = static_cast<std::uint64_t>(requested_size);
        check_status(shmem_uint64_min_reduce(SHMEM_TEAM_WORLD,
                                             &scratch[scratch_staging_min],
                                             &scratch[scratch_staging_request],
                                             1),
                     "OSHMPI staging-size minimum agreement");
        check_status(shmem_uint64_max_reduce(SHMEM_TEAM_WORLD,
                                             &scratch[scratch_staging_max],
                                             &scratch[scratch_staging_request],
                                             1),
                     "OSHMPI staging-size maximum agreement");
        CCL_THROW_IF_NOT(scratch[scratch_staging_min] == scratch[scratch_staging_max],
                         "CCL_OSHMPI_STAGING_SIZE must be identical on every PE");

        lane_size = (requested_size / 2 / staging_alignment) * staging_alignment;
        CCL_THROW_IF_NOT(lane_size >= staging_alignment,
                         "CCL_OSHMPI_STAGING_SIZE must provide two non-empty lanes");

        staging = static_cast<char*>(shmem_malloc(checked_multiply(lane_size, 2, "staging")));
        CCL_THROW_IF_NOT(staging,
                         "shmem_malloc failed for OSHMPI staging; increase SHMEM_SYMMETRIC_SIZE");

        users = 1;
        LOG_INFO("OSHMPI runtime initialized: rank ",
                 world_rank,
                 "/",
                 world_size,
                 ", staging bytes ",
                 lane_size * 2);
    }
    catch (...) {
        if (staging) {
            shmem_free(staging);
            staging = nullptr;
        }
        if (scratch) {
            shmem_free(scratch);
            scratch = nullptr;
        }
        if (initialized) {
            shmem_finalize();
        }
        initialized = false;
        world_rank = -1;
        world_size = 0;
        lane_size = 0;
        throw;
    }
}

void oshmpi_runtime::release() noexcept {
    std::lock_guard<std::mutex> state_lock(state_mutex);
    if (users == 0) {
        return;
    }

    --users;
    if (users > 0) {
        return;
    }

    std::lock_guard<std::mutex> operation_lock(operation_mutex);
    if (staging) {
        shmem_free(staging);
        staging = nullptr;
    }
    if (scratch) {
        shmem_free(scratch);
        scratch = nullptr;
    }
    if (initialized) {
        shmem_finalize();
    }

    initialized = false;
    finalized = true;
    world_rank = -1;
    world_size = 0;
    lane_size = 0;
}

void oshmpi_runtime::check_ready() const {
    CCL_THROW_IF_NOT(initialized && staging, "OSHMPI runtime is not initialized");
}

void oshmpi_runtime::barrier() {
    std::lock_guard<std::mutex> lock(operation_mutex);
    check_ready();
    shmem_barrier_all();
}

void oshmpi_runtime::allgather(const void* send_buf, void* recv_buf, std::size_t bytes) {
    std::lock_guard<std::mutex> lock(operation_mutex);
    check_ready();
    if (bytes == 0) {
        return;
    }

    checked_multiply(bytes, static_cast<std::size_t>(world_size), "allgather");
    const std::size_t max_chunk =
        std::min(lane_size / static_cast<std::size_t>(world_size),
                 static_cast<std::size_t>(INT_MAX));
    CCL_THROW_IF_NOT(max_chunk > 0, "OSHMPI staging is too small for allgather");

    const char* source = static_cast<const char*>(send_buf);
    char* destination = static_cast<char*>(recv_buf);
    char* source_stage = staging;
    char* destination_stage = staging + lane_size;

    for (std::size_t offset = 0; offset < bytes;) {
        const std::size_t chunk = std::min(bytes - offset, max_chunk);
        std::memcpy(source_stage, source + offset, chunk);
        check_status(shmem_fcollectmem(
                         SHMEM_TEAM_WORLD, destination_stage, source_stage, chunk),
                     "shmem_fcollectmem");
        for (int peer = 0; peer < world_size; ++peer) {
            std::memcpy(destination + static_cast<std::size_t>(peer) * bytes + offset,
                        destination_stage + static_cast<std::size_t>(peer) * chunk,
                        chunk);
        }
        offset += chunk;
    }
}

void oshmpi_runtime::allgather(const void* send_buf,
                               const ccl::vector_class<void*>& recv_bufs,
                               std::size_t bytes) {
    std::lock_guard<std::mutex> lock(operation_mutex);
    check_ready();
    CCL_THROW_IF_NOT(recv_bufs.size() == static_cast<std::size_t>(world_size),
                     "allgather receive buffer count must match communicator size");
    if (bytes == 0) {
        return;
    }

    const std::size_t max_chunk =
        std::min(lane_size / static_cast<std::size_t>(world_size),
                 static_cast<std::size_t>(INT_MAX));
    CCL_THROW_IF_NOT(max_chunk > 0, "OSHMPI staging is too small for allgather");

    const char* source = static_cast<const char*>(send_buf);
    char* source_stage = staging;
    char* destination_stage = staging + lane_size;

    for (std::size_t offset = 0; offset < bytes;) {
        const std::size_t chunk = std::min(bytes - offset, max_chunk);
        std::memcpy(source_stage, source + offset, chunk);
        check_status(shmem_fcollectmem(
                         SHMEM_TEAM_WORLD, destination_stage, source_stage, chunk),
                     "shmem_fcollectmem");
        for (int peer = 0; peer < world_size; ++peer) {
            char* destination = static_cast<char*>(recv_bufs[static_cast<std::size_t>(peer)]);
            std::memcpy(destination + offset,
                        destination_stage + static_cast<std::size_t>(peer) * chunk,
                        chunk);
        }
        offset += chunk;
    }
}

void oshmpi_runtime::allreduce(const void* send_buf,
                               void* recv_buf,
                               std::size_t count,
                               ccl::datatype dtype,
                               ccl::reduction reduction) {
    std::lock_guard<std::mutex> lock(operation_mutex);
    check_ready();
    if (count == 0) {
        return;
    }

    const std::size_t datatype_size = ccl::get_datatype_size(dtype);
    const std::size_t max_chunk =
        std::min(lane_size / datatype_size, static_cast<std::size_t>(INT_MAX));
    CCL_THROW_IF_NOT(max_chunk > 0, "OSHMPI staging is too small for allreduce datatype");

    const char* source = static_cast<const char*>(send_buf);
    char* destination = static_cast<char*>(recv_buf);
    char* source_stage = staging;
    char* destination_stage = staging + lane_size;

    for (std::size_t offset = 0; offset < count;) {
        const std::size_t chunk = std::min(count - offset, max_chunk);
        const std::size_t chunk_bytes = checked_multiply(chunk, datatype_size, "allreduce");
        std::memcpy(source_stage, source + offset * datatype_size, chunk_bytes);
        reduce_chunk(destination_stage, source_stage, chunk, dtype, reduction);
        std::memcpy(destination + offset * datatype_size, destination_stage, chunk_bytes);
        offset += chunk;
    }
}

void oshmpi_runtime::alltoall(const void* send_buf,
                              void* recv_buf,
                              std::size_t bytes_per_peer) {
    std::lock_guard<std::mutex> lock(operation_mutex);
    check_ready();
    if (bytes_per_peer == 0) {
        return;
    }

    checked_multiply(bytes_per_peer, static_cast<std::size_t>(world_size), "alltoall");
    const std::size_t max_chunk =
        std::min(lane_size / static_cast<std::size_t>(world_size),
                 static_cast<std::size_t>(INT_MAX));
    CCL_THROW_IF_NOT(max_chunk > 0, "OSHMPI staging is too small for alltoall");

    const char* source = static_cast<const char*>(send_buf);
    char* destination = static_cast<char*>(recv_buf);
    char* source_stage = staging;
    char* destination_stage = staging + lane_size;

    for (std::size_t offset = 0; offset < bytes_per_peer;) {
        const std::size_t chunk = std::min(bytes_per_peer - offset, max_chunk);
        for (int peer = 0; peer < world_size; ++peer) {
            std::memcpy(source_stage + static_cast<std::size_t>(peer) * chunk,
                        source + static_cast<std::size_t>(peer) * bytes_per_peer + offset,
                        chunk);
        }
        check_status(shmem_alltoallmem(
                         SHMEM_TEAM_WORLD, destination_stage, source_stage, chunk),
                     "shmem_alltoallmem");
        for (int peer = 0; peer < world_size; ++peer) {
            std::memcpy(destination + static_cast<std::size_t>(peer) * bytes_per_peer + offset,
                        destination_stage + static_cast<std::size_t>(peer) * chunk,
                        chunk);
        }
        offset += chunk;
    }
}

void oshmpi_runtime::broadcast(const void* send_buf,
                               void* recv_buf,
                               std::size_t bytes,
                               int root) {
    std::lock_guard<std::mutex> lock(operation_mutex);
    check_ready();
    CCL_THROW_IF_NOT(root >= 0 && root < world_size, "invalid broadcast root: ", root);
    if (bytes == 0) {
        return;
    }

    const char* source = static_cast<const char*>(send_buf);
    char* destination = static_cast<char*>(recv_buf);
    char* source_stage = staging;
    char* destination_stage = staging + lane_size;

    for (std::size_t offset = 0; offset < bytes;) {
        const std::size_t chunk =
            std::min(bytes - offset,
                     std::min(lane_size, static_cast<std::size_t>(INT_MAX)));
        if (world_rank == root) {
            std::memcpy(source_stage, source + offset, chunk);
        }
        check_status(shmem_broadcastmem(
                         SHMEM_TEAM_WORLD, destination_stage, source_stage, chunk, root),
                     "shmem_broadcastmem");
        std::memcpy(destination + offset, destination_stage, chunk);
        offset += chunk;
    }
}

void oshmpi_runtime::reduce_chunk(void* destination,
                                  const void* source,
                                  std::size_t count,
                                  ccl::datatype dtype,
                                  ccl::reduction reduction) {
#define CCL_OSHMPI_REDUCE_CASE(ccl_dtype, c_type, shmem_type) \
    case ccl_dtype: { \
        int status = SHMEM_OTHER_ERR; \
        switch (reduction) { \
            case ccl::reduction::sum: \
                status = shmem_##shmem_type##_sum_reduce( \
                    SHMEM_TEAM_WORLD, \
                    static_cast<c_type*>(destination), \
                    static_cast<const c_type*>(source), \
                    count); \
                break; \
            case ccl::reduction::prod: \
                status = shmem_##shmem_type##_prod_reduce( \
                    SHMEM_TEAM_WORLD, \
                    static_cast<c_type*>(destination), \
                    static_cast<const c_type*>(source), \
                    count); \
                break; \
            case ccl::reduction::min: \
                status = shmem_##shmem_type##_min_reduce( \
                    SHMEM_TEAM_WORLD, \
                    static_cast<c_type*>(destination), \
                    static_cast<const c_type*>(source), \
                    count); \
                break; \
            case ccl::reduction::max: \
                status = shmem_##shmem_type##_max_reduce( \
                    SHMEM_TEAM_WORLD, \
                    static_cast<c_type*>(destination), \
                    static_cast<const c_type*>(source), \
                    count); \
                break; \
            default: \
                CCL_THROW("unsupported OSHMPI reduction: ", static_cast<int>(reduction)); \
        } \
        check_status(status, "OSHMPI reduction"); \
        return; \
    }

    switch (dtype) {
        CCL_OSHMPI_REDUCE_CASE(ccl::datatype::int8, std::int8_t, int8)
        CCL_OSHMPI_REDUCE_CASE(ccl::datatype::uint8, std::uint8_t, uint8)
        CCL_OSHMPI_REDUCE_CASE(ccl::datatype::int16, std::int16_t, int16)
        CCL_OSHMPI_REDUCE_CASE(ccl::datatype::uint16, std::uint16_t, uint16)
        CCL_OSHMPI_REDUCE_CASE(ccl::datatype::int32, std::int32_t, int32)
        CCL_OSHMPI_REDUCE_CASE(ccl::datatype::uint32, std::uint32_t, uint32)
        CCL_OSHMPI_REDUCE_CASE(ccl::datatype::int64, std::int64_t, int64)
        CCL_OSHMPI_REDUCE_CASE(ccl::datatype::uint64, std::uint64_t, uint64)
        CCL_OSHMPI_REDUCE_CASE(ccl::datatype::float32, float, float)
        CCL_OSHMPI_REDUCE_CASE(ccl::datatype::float64, double, double)
        default:
            CCL_THROW("unsupported OSHMPI allreduce datatype: ", static_cast<int>(dtype));
    }

#undef CCL_OSHMPI_REDUCE_CASE
}

} // namespace ccl

#endif // CCL_ENABLE_OSHMPI
