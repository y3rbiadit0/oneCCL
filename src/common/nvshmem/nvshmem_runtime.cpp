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

#include "common/nvshmem/nvshmem_runtime.hpp"

#include "oneapi/ccl.hpp"
#include "common/log/log.hpp"
#include "common/nvshmem/nvshmem_adapter.h"
#include "common/utils/sycl_utils.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
#error "CCL_ENABLE_NVSHMEM requires SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND"
#endif

namespace ccl::nvshmem {
namespace {

constexpr size_t default_arena_bytes = 64 * 1024 * 1024;
constexpr size_t metadata_bytes = 4096;
constexpr size_t signal_bytes = 4096;
constexpr size_t lane_alignment = 256;
constexpr size_t kvs_payload_chars = 120;
constexpr const char* staging_size_env = "CCL_NVSHMEM_STAGING_SIZE";
constexpr std::chrono::seconds kvs_timeout{ 120 };

class staged_command_status {
public:
    void record_error(const char* operation) noexcept {
        if (failed.load(std::memory_order_relaxed)) {
            return;
        }
        const char* detail = oneccl_nvshmem_last_error();
        std::snprintf(error,
                      sizeof(error),
                      "%s failed%s%s",
                      operation,
                      (detail != nullptr && detail[0] != '\0') ? ": " : "",
                      (detail != nullptr) ? detail : "");
        failed.store(true, std::memory_order_release);
    }

    void record_unknown_error() noexcept {
        if (failed.load(std::memory_order_relaxed)) {
            return;
        }
        std::snprintf(error, sizeof(error), "staged native command threw an exception");
        failed.store(true, std::memory_order_release);
    }

    void throw_if_failed() const {
        if (failed.load(std::memory_order_acquire)) {
            CCL_THROW(error);
        }
    }

private:
    std::atomic<bool> failed{ false };
    char error[512]{};
};

void check_adapter(int status, const char* operation) {
    CCL_THROW_IF_NOT(status == 0,
                     operation,
                     " failed: ",
                     oneccl_nvshmem_last_error());
}

size_t checked_staging_size() {
    const char* configured = std::getenv(staging_size_env);
    if (configured == nullptr || configured[0] == '\0') {
        return default_arena_bytes;
    }

    const std::string value(configured);
    size_t index = 0;
    size_t parsed = 0;
    while (index < value.size() && std::isdigit(static_cast<unsigned char>(value[index]))) {
        const unsigned int digit = static_cast<unsigned int>(value[index] - '0');
        CCL_THROW_IF_NOT(parsed <= (std::numeric_limits<size_t>::max() - digit) / 10,
                         staging_size_env,
                         " overflows size_t: ",
                         value);
        parsed = parsed * 10 + digit;
        index++;
    }
    CCL_THROW_IF_NOT(index > 0, staging_size_env, " has no numeric value: ", value);

    std::string suffix = value.substr(index);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });

    size_t multiplier = 1;
    if (suffix == "k" || suffix == "kb") {
        multiplier = 1024;
    }
    else if (suffix == "m" || suffix == "mb") {
        multiplier = 1024 * 1024;
    }
    else if (suffix == "g" || suffix == "gb") {
        multiplier = 1024ULL * 1024 * 1024;
    }
    else {
        CCL_THROW_IF_NOT(suffix.empty() || suffix == "b",
                         staging_size_env,
                         " has an unsupported suffix: ",
                         value);
    }

    CCL_THROW_IF_NOT(parsed > 0, staging_size_env, " must be positive");
    CCL_THROW_IF_NOT(parsed <= std::numeric_limits<size_t>::max() / multiplier,
                     staging_size_env,
                     " overflows size_t: ",
                     value);
    return parsed * multiplier;
}

std::string encode_uid(const std::vector<unsigned char>& uid) {
    static constexpr char digits[] = "0123456789abcdef";
    CCL_THROW_IF_NOT(uid.size() <= (std::numeric_limits<size_t>::max() - 4) / 2,
                     "NVSHMEM UID is too large to encode");
    std::string encoded = "UID:";
    encoded.reserve(4 + uid.size() * 2);
    for (unsigned char byte : uid) {
        encoded.push_back(digits[byte >> 4]);
        encoded.push_back(digits[byte & 0x0f]);
    }
    return encoded;
}

int hex_value(char character) {
    if (character >= '0' && character <= '9') {
        return character - '0';
    }
    if (character >= 'a' && character <= 'f') {
        return character - 'a' + 10;
    }
    if (character >= 'A' && character <= 'F') {
        return character - 'A' + 10;
    }
    return -1;
}

std::vector<unsigned char> decode_uid(const std::string& value, size_t expected_size) {
    CCL_THROW_IF_NOT(value.find('\0') == std::string::npos,
                     "NVSHMEM UID KVS value contains an interior terminator");
    CCL_THROW_IF_NOT(value.rfind("ERROR:", 0) != 0,
                     "NVSHMEM UID creation failed on rank zero: ",
                     value.substr(6));
    CCL_THROW_IF_NOT(value.rfind("UID:", 0) == 0, "invalid NVSHMEM UID KVS value");
    CCL_THROW_IF_NOT(value.size() == 4 + expected_size * 2,
                     "NVSHMEM UID KVS value has an unexpected size");

    std::vector<unsigned char> uid(expected_size);
    for (size_t index = 0; index < expected_size; index++) {
        const int high = hex_value(value[4 + index * 2]);
        const int low = hex_value(value[5 + index * 2]);
        CCL_THROW_IF_NOT(high >= 0 && low >= 0, "NVSHMEM UID KVS value is not hexadecimal");
        uid[index] = static_cast<unsigned char>((high << 4) | low);
    }
    return uid;
}

std::string encode_error(const std::string& error) {
    std::string value = "ERROR:";
    const std::string diagnostic = error.empty() ? "unknown" : error;
    value.append(diagnostic, 0, kvs_payload_chars - value.size());
    return value;
}

std::vector<char> encode_string(const std::string& value) {
    CCL_THROW_IF_NOT(value.size() <= kvs_payload_chars,
                     "NVSHMEM KVS value exceeds the transport limit");
    CCL_THROW_IF_NOT(value.find('\0') == std::string::npos,
                     "NVSHMEM KVS value contains an interior terminator");
    std::vector<char> encoded(value.begin(), value.end());
    encoded.push_back('\0');
    return encoded;
}

std::string wait_kvs_value(const std::shared_ptr<ccl::v1::kvs_interface>& kvs,
                           const std::string& key) {
    const auto deadline = std::chrono::steady_clock::now() + kvs_timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        const ccl::vector_class<char> encoded = kvs->get(key);
        if (!encoded.empty()) {
            const auto terminator = std::find(encoded.begin(), encoded.end(), '\0');
            CCL_THROW_IF_NOT(terminator != encoded.end(),
                             "NVSHMEM KVS value is not null-terminated for key ",
                             key);
            return std::string(encoded.begin(), terminator);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CCL_THROW("timed out waiting for NVSHMEM KVS key ", key);
}

void publish_chunked_value(const std::shared_ptr<ccl::v1::kvs_interface>& kvs,
                           const std::string& key,
                           const std::string& value) {
    size_t chunk = 0;
    for (size_t offset = 0; offset < value.size(); offset += kvs_payload_chars, chunk++) {
        kvs->set(key + "_CHUNK_" + std::to_string(chunk),
                 encode_string(value.substr(offset, kvs_payload_chars)));
    }
    kvs->set(key + "_STATUS", encode_string("OK:" + std::to_string(value.size())));
}

std::string wait_chunked_value(const std::shared_ptr<ccl::v1::kvs_interface>& kvs,
                               const std::string& key,
                               size_t expected_size) {
    const std::string status = wait_kvs_value(kvs, key + "_STATUS");
    CCL_THROW_IF_NOT(status.rfind("ERROR:", 0) != 0,
                     "NVSHMEM UID creation failed on rank zero: ",
                     status.substr(6));
    CCL_THROW_IF_NOT(status == "OK:" + std::to_string(expected_size),
                     "invalid NVSHMEM UID KVS status");

    std::string value;
    value.reserve(expected_size);
    size_t chunk = 0;
    for (size_t offset = 0; offset < expected_size; offset += kvs_payload_chars, chunk++) {
        const std::string part =
            wait_kvs_value(kvs, key + "_CHUNK_" + std::to_string(chunk));
        const size_t expected_chunk_size =
            std::min(kvs_payload_chars, expected_size - offset);
        CCL_THROW_IF_NOT(part.size() == expected_chunk_size,
                         "NVSHMEM UID KVS chunk has an unexpected size");
        value += part;
    }
    return value;
}

std::string agree_phase(const std::shared_ptr<ccl::v1::kvs_interface>& kvs,
                        const std::string& prefix,
                        int rank,
                        int size,
                        const std::string& local_error,
                        const std::string& success_value) {
    std::string local_value = local_error.empty() ? "OK:" + success_value
                                                  : encode_error(local_error);
    if (local_value.size() > kvs_payload_chars) {
        local_value = encode_error("agreement success state exceeds the KVS transport limit");
    }
    kvs->set(prefix + "_" + std::to_string(rank), encode_string(local_value));

    std::string phase_error;
    for (int peer = 0; peer < size; peer++) {
        const std::string value = wait_kvs_value(kvs, prefix + "_" + std::to_string(peer));
        if (value.rfind("ERROR:", 0) == 0) {
            if (phase_error.empty()) {
                phase_error = "rank " + std::to_string(peer) + ": " + value.substr(6);
            }
        }
        else if (value != "OK:" + success_value && phase_error.empty()) {
            phase_error = "rank " + std::to_string(peer) + " reported '" + value +
                          "', expected 'OK:" + success_value + "'";
        }
    }
    return phase_error;
}

void validate_user_pointer(const void* pointer,
                           size_t bytes,
                           const sycl::context& context,
                           const char* name) {
    if (bytes == 0) {
        return;
    }
    CCL_THROW_IF_NOT(pointer != nullptr, name, " is null for a non-empty NVSHMEM operation");
    const sycl::usm::alloc allocation = sycl::get_pointer_type(pointer, context);
    CCL_THROW_IF_NOT(allocation == sycl::usm::alloc::device ||
                         allocation == sycl::usm::alloc::shared,
                     name,
                     " must be device or shared USM");
}

} // namespace

runtime::~runtime() noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex);
        finalize_locked();
    }
    catch (const std::exception& error) {
        LOG_WARN("NVSHMEM runtime finalization failed: ", error.what());
    }
    catch (...) {
        LOG_WARN("NVSHMEM runtime finalization failed with an unknown error");
    }
}

void runtime::initialize(const sycl::device& requested_device,
                         const sycl::context& requested_context,
                         int rank,
                         int size,
                         const std::shared_ptr<ccl::v1::kvs_interface>& kvs) {
    std::lock_guard<std::mutex> lock(mutex);
    if (current_state == state::ready) {
        validate_existing_locked(requested_device, requested_context, rank, size);
        communicator_count++;
        return;
    }
    CCL_THROW_IF_NOT(current_state == state::uninitialized,
                     "NVSHMEM runtime is not available after a prior initialization failure");

    current_state = state::initializing;
    try {
        initialize_locked(requested_device, requested_context, rank, size, kvs);
        current_state = state::ready;
        communicator_count = 1;
    }
    catch (...) {
        if (owns_initialization || arena != nullptr) {
            // The failing PE may not have peers available for collective cleanup.
            owns_initialization = false;
            arena = nullptr;
        }
        finalize_locked();
        current_state = state::failed;
        throw;
    }
}

void runtime::initialize_locked(const sycl::device& requested_device,
                                const sycl::context& requested_context,
                                int rank,
                                int size,
                                const std::shared_ptr<ccl::v1::kvs_interface>& kvs) {
    CCL_THROW_IF_NOT(kvs, "NVSHMEM runtime requires a KVS instance");
    CCL_THROW_IF_NOT(rank >= 0 && size > 0 && rank < size, "invalid NVSHMEM runtime topology");

    key_prefix = "ONECCL_NVSHMEM_V1_" + std::to_string(size);
    int requested_cuda_device = -1;
    bool externally_initialized = false;
    std::string preflight_error;
    try {
        const auto native_device =
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(requested_device);
        requested_cuda_device = static_cast<int>(native_device);
        arena_bytes = checked_staging_size();
        externally_initialized = oneccl_nvshmem_is_initialized() != 0;
        if (externally_initialized) {
            int current_device = -1;
            check_adapter(oneccl_nvshmem_cuda_get_device(&current_device), "cudaGetDevice");
            CCL_THROW_IF_NOT(current_device == requested_cuda_device,
                             "externally initialized NVSHMEM uses CUDA device ",
                             current_device,
                             ", requested device is ",
                             requested_cuda_device);
        }
        else {
            check_adapter(oneccl_nvshmem_cuda_set_device(requested_cuda_device), "cudaSetDevice");
        }
    }
    catch (const std::exception& error) {
        preflight_error = error.what();
    }

    const std::string preflight_value = std::to_string(arena_bytes) + ":" +
                                        (externally_initialized ? "external" : "owned");
    const std::string preflight_phase_error = agree_phase(kvs,
                                                          key_prefix + "_PREFLIGHT",
                                                          rank,
                                                          size,
                                                          preflight_error,
                                                          preflight_value);
    CCL_THROW_IF_NOT(preflight_phase_error.empty(),
                     "NVSHMEM preflight failed: ",
                     preflight_phase_error);

    if (!externally_initialized) {
        const size_t uid_size = oneccl_nvshmem_uid_size();
        CCL_THROW_IF_NOT(uid_size > 0, "NVSHMEM reported an empty UID type");
        CCL_THROW_IF_NOT(uid_size <= (std::numeric_limits<size_t>::max() - 4) / 2,
                         "NVSHMEM UID is too large to transport");
        const size_t encoded_uid_size = 4 + uid_size * 2;
        const std::string uid_key = key_prefix + "_UID";
        if (rank == 0) {
            std::vector<unsigned char> root_uid(uid_size);
            if (oneccl_nvshmem_get_uid(root_uid.data(), root_uid.size()) == 0) {
                try {
                    publish_chunked_value(kvs, uid_key, encode_uid(root_uid));
                }
                catch (const std::exception& error) {
                    kvs->set(uid_key + "_STATUS", encode_string(encode_error(error.what())));
                }
            }
            else {
                kvs->set(uid_key + "_STATUS",
                         encode_string(encode_error(oneccl_nvshmem_last_error())));
            }
        }

        std::vector<unsigned char> uid;
        std::string uid_error;
        try {
            uid = decode_uid(wait_chunked_value(kvs, uid_key, encoded_uid_size), uid_size);
        }
        catch (const std::exception& error) {
            uid_error = error.what();
        }
        const std::string uid_phase_error =
            agree_phase(kvs, key_prefix + "_UID_READY", rank, size, uid_error, "ready");
        CCL_THROW_IF_NOT(uid_phase_error.empty(), "NVSHMEM UID exchange failed: ", uid_phase_error);

        std::string init_error;
        if (oneccl_nvshmem_hostlib_init(rank, size, uid.data(), uid.size()) == 0) {
            owns_initialization = true;
        }
        else {
            init_error = oneccl_nvshmem_last_error();
        }
        const std::string init_phase_error =
            agree_phase(kvs, key_prefix + "_INIT", rank, size, init_error, "ready");
        if (!init_phase_error.empty()) {
            // A PE that failed initialization cannot participate in collective cleanup.
            owns_initialization = false;
            CCL_THROW("NVSHMEM initialization failed: ", init_phase_error);
        }
    }

    std::string topology_error;
    try {
        CCL_THROW_IF_NOT(oneccl_nvshmem_is_initialized(),
                         "NVSHMEM device initialization is incomplete");
        CCL_THROW_IF_NOT(oneccl_nvshmem_my_pe() == rank,
                         "oneCCL rank does not match the NVSHMEM PE rank");
        CCL_THROW_IF_NOT(oneccl_nvshmem_n_pes() == size,
                         "oneCCL size does not match the NVSHMEM PE count");
    }
    catch (const std::exception& error) {
        topology_error = error.what();
    }
    const std::string topology_phase_error =
        agree_phase(kvs, key_prefix + "_TOPOLOGY", rank, size, topology_error, "ready");
    CCL_THROW_IF_NOT(topology_phase_error.empty(),
                     "NVSHMEM topology validation failed: ",
                     topology_phase_error);

    pe_rank = rank;
    pe_count = size;
    cuda_device = requested_cuda_device;
    device = requested_device;
    context = requested_context;

    std::string arena_error;
    try {
        allocate_arena_locked();
    }
    catch (const std::exception& error) {
        arena_error = error.what();
    }
    const std::string arena_phase_error =
        agree_phase(kvs, key_prefix + "_ARENA", rank, size, arena_error, "ready");
    if (!arena_phase_error.empty()) {
        // Symmetric allocation failure may be asymmetric; avoid unmatched free/finalize calls.
        arena = nullptr;
        owns_initialization = false;
        CCL_THROW("NVSHMEM arena allocation failed: ", arena_phase_error);
    }
    bootstrap_kvs = kvs;
}

void runtime::release() noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex);
        if (current_state != state::ready) {
            return;
        }
        if (communicator_count == 0) {
            LOG_WARN("NVSHMEM communicator release was called without an active communicator");
            return;
        }
        communicator_count--;
        if (communicator_count != 0) {
            return;
        }

        try {
            CCL_THROW_IF_NOT(bootstrap_kvs, "NVSHMEM shutdown KVS is unavailable");
            const std::string shutdown_error = agree_phase(bootstrap_kvs,
                                                           key_prefix + "_SHUTDOWN",
                                                           pe_rank,
                                                           pe_count,
                                                           {},
                                                           "ready");
            CCL_THROW_IF_NOT(shutdown_error.empty(),
                             "NVSHMEM shutdown agreement failed: ",
                             shutdown_error);
        }
        catch (...) {
            current_state = state::failed;
            throw;
        }
        finalize_locked();
    }
    catch (const std::exception& error) {
        LOG_WARN("NVSHMEM communicator release failed: ", error.what());
    }
    catch (...) {
        LOG_WARN("NVSHMEM communicator release failed with an unknown error");
    }
}

void runtime::allocate_arena_locked() {
    CCL_THROW_IF_NOT(arena_bytes > metadata_bytes + signal_bytes,
                     staging_size_env,
                     " is too small for staging metadata");

    const size_t payload_bytes = arena_bytes - metadata_bytes - signal_bytes;
    lane_bytes = (payload_bytes / 2 / lane_alignment) * lane_alignment;
    CCL_THROW_IF_NOT(lane_bytes > 0,
                     staging_size_env,
                     " is too small for source and destination lanes");

    arena = oneccl_nvshmem_malloc(arena_bytes);
    CCL_THROW_IF_NOT(arena != nullptr,
                     "NVSHMEM symmetric arena allocation failed: ",
                     oneccl_nvshmem_last_error());

    auto* base = static_cast<unsigned char*>(arena);
    source_lane = base;
    destination_lane = base + lane_bytes;
    metadata_region = base + lane_bytes * 2;
    signal_region = static_cast<unsigned char*>(metadata_region) + metadata_bytes;
}

void runtime::validate_existing_locked(const sycl::device& requested_device,
                                       const sycl::context& requested_context,
                                       int rank,
                                       int size) const {
    CCL_THROW_IF_NOT(device && context, "NVSHMEM runtime metadata is incomplete");
    CCL_THROW_IF_NOT(*device == requested_device && *context == requested_context,
                     "all NVSHMEM communicators in a process must use the same CUDA device and context");
    CCL_THROW_IF_NOT(pe_rank == rank && pe_count == size,
                     "all NVSHMEM communicators in a process must use the same topology");
}

sycl::event runtime::submit_staged(const void* source,
                                   void* destination,
                                   size_t bytes,
                                   sycl::queue& queue,
                                   const staged_operation& operation) {
    std::lock_guard<std::mutex> lock(mutex);
    CCL_THROW_IF_NOT(current_state == state::ready, "NVSHMEM runtime is not initialized");
    CCL_THROW_IF_NOT(operation, "NVSHMEM staged operation callback is empty");
    CCL_THROW_IF_NOT(queue.get_backend() == sycl::backend::ext_oneapi_cuda,
                     "NVSHMEM staging requires a CUDA-backed SYCL queue");
    CCL_THROW_IF_NOT(queue.is_in_order(), "NVSHMEM staging requires an in-order SYCL queue");
    CCL_THROW_IF_NOT(device && context && queue.get_device() == *device &&
                         queue.get_context() == *context,
                     "NVSHMEM staging queue does not match the communicator device and context");

    validate_user_pointer(source, bytes, *context, "NVSHMEM source buffer");
    validate_user_pointer(destination, bytes, *context, "NVSHMEM destination buffer");
    if (bytes == 0) {
        return ccl::utils::submit_barrier(queue);
    }

    const staged_operation staged_operation_copy = operation;
    void* const source_lane_copy = source_lane;
    void* const destination_lane_copy = destination_lane;
    const size_t lane_bytes_copy = lane_bytes;
    const int cuda_device_copy = cuda_device;

    const auto command_status = std::make_shared<staged_command_status>();
    sycl::event native_completion = queue.submit([&](sycl::handler& handler) {
        if (last_completion) {
            handler.depends_on(*last_completion);
        }
        handler.ext_codeplay_enqueue_native_command(
            [source,
             destination,
             bytes,
             source_lane_copy,
             destination_lane_copy,
             lane_bytes_copy,
             cuda_device_copy,
             staged_operation_copy,
             command_status](sycl::interop_handle interop) {
                try {
                    auto native_stream =
                        interop.get_native_queue<sycl::backend::ext_oneapi_cuda>();
                    void* stream = reinterpret_cast<void*>(native_stream);
                    if (oneccl_nvshmem_cuda_set_device(cuda_device_copy) != 0) {
                        command_status->record_error("cudaSetDevice");
                        return;
                    }

                    size_t offset = 0;
                    while (offset < bytes) {
                        const size_t chunk_bytes = std::min(bytes - offset, lane_bytes_copy);
                        const auto* source_bytes =
                            static_cast<const unsigned char*>(source) + offset;
                        auto* destination_bytes =
                            static_cast<unsigned char*>(destination) + offset;

                        if (oneccl_nvshmem_copy_async(
                                source_lane_copy, source_bytes, chunk_bytes, stream) != 0) {
                            command_status->record_error("staging source copy");
                            return;
                        }
                        if (staged_operation_copy(destination_lane_copy,
                                                  source_lane_copy,
                                                  chunk_bytes,
                                                  stream) != 0) {
                            command_status->record_error("staged operation");
                            return;
                        }
                        if (oneccl_nvshmem_copy_async(destination_bytes,
                                                      destination_lane_copy,
                                                      chunk_bytes,
                                                      stream) != 0) {
                            command_status->record_error("staging destination copy");
                            return;
                        }
                        offset += chunk_bytes;
                    }
                }
                catch (...) {
                    command_status->record_unknown_error();
                }
            });
    });
    last_completion = native_completion;
    sycl::event completion;
    try {
        completion = queue.submit([&](sycl::handler& handler) {
            handler.depends_on(native_completion);
            handler.host_task([command_status]() {
                command_status->throw_if_failed();
            });
        });
    }
    catch (...) {
        native_completion.wait_and_throw();
        throw;
    }
    return completion;
}

size_t runtime::staging_lane_size() const {
    std::lock_guard<std::mutex> lock(mutex);
    CCL_THROW_IF_NOT(current_state == state::ready, "NVSHMEM runtime is not initialized");
    return lane_bytes;
}

bool runtime::is_ready() const {
    std::lock_guard<std::mutex> lock(mutex);
    return current_state == state::ready;
}

void runtime::validate_staging(sycl::queue& queue, bool enabled) {
    const std::string mode_phase_error = agree_phase(bootstrap_kvs,
                                                     key_prefix + "_STAGING_MODE",
                                                     pe_rank,
                                                     pe_count,
                                                     {},
                                                     enabled ? "enabled" : "disabled");
    CCL_THROW_IF_NOT(mode_phase_error.empty(),
                     "NVSHMEM staging validation mode differs across ranks: ",
                     mode_phase_error);
    if (!enabled) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        if (staging_validation_completed) {
            return;
        }
    }

    std::string validation_error;
    try {
        const staged_operation identity = [](void* destination,
                                             const void* source,
                                             size_t bytes,
                                             void* native_stream) {
            return oneccl_nvshmem_copy_async(destination, source, bytes, native_stream);
        };

        const size_t lane_size = staging_lane_size();
        const std::vector<size_t> sizes{ lane_size - 1, lane_size, lane_size + 1 };
        for (size_t bytes : sizes) {
            std::vector<unsigned char> expected(bytes);
            for (size_t index = 0; index < bytes; index++) {
                expected[index] = static_cast<unsigned char>((index + pe_rank) % 251);
            }

            unsigned char* shared_source = sycl::malloc_shared<unsigned char>(bytes, queue);
            unsigned char* shared_destination =
                sycl::malloc_shared<unsigned char>(bytes, queue);
            if (shared_source == nullptr || shared_destination == nullptr) {
                if (shared_source != nullptr) {
                    sycl::free(shared_source, queue);
                }
                if (shared_destination != nullptr) {
                    sycl::free(shared_destination, queue);
                }
                CCL_THROW("failed to allocate shared USM for NVSHMEM staging validation");
            }
            std::copy(expected.begin(), expected.end(), shared_source);
            std::fill(shared_destination, shared_destination + bytes, 0);
            try {
                submit_staged(shared_source, shared_destination, bytes, queue, identity)
                    .wait_and_throw();
                const auto mismatch = std::mismatch(expected.begin(),
                                                    expected.end(),
                                                    shared_destination);
                CCL_THROW_IF_NOT(
                    mismatch.first == expected.end(),
                    "shared USM NVSHMEM staging validation failed at byte ",
                    std::distance(expected.begin(), mismatch.first),
                    ": expected ",
                    static_cast<unsigned int>(*mismatch.first),
                    ", received ",
                    static_cast<unsigned int>(*mismatch.second));
            }
            catch (...) {
                sycl::free(shared_source, queue);
                sycl::free(shared_destination, queue);
                throw;
            }
            sycl::free(shared_source, queue);
            sycl::free(shared_destination, queue);

            unsigned char* device_source = sycl::malloc_device<unsigned char>(bytes, queue);
            unsigned char* device_destination =
                sycl::malloc_device<unsigned char>(bytes, queue);
            if (device_source == nullptr || device_destination == nullptr) {
                if (device_source != nullptr) {
                    sycl::free(device_source, queue);
                }
                if (device_destination != nullptr) {
                    sycl::free(device_destination, queue);
                }
                CCL_THROW("failed to allocate device USM for NVSHMEM staging validation");
            }
            std::vector<unsigned char> actual(bytes, 0);
            try {
                queue.memcpy(device_source, expected.data(), bytes).wait_and_throw();
                submit_staged(device_source, device_destination, bytes, queue, identity)
                    .wait_and_throw();
                queue.memcpy(actual.data(), device_destination, bytes).wait_and_throw();
                const auto mismatch =
                    std::mismatch(expected.begin(), expected.end(), actual.begin());
                CCL_THROW_IF_NOT(
                    mismatch.first == expected.end(),
                    "device USM NVSHMEM staging validation failed at byte ",
                    std::distance(expected.begin(), mismatch.first),
                    ": expected ",
                    static_cast<unsigned int>(*mismatch.first),
                    ", received ",
                    static_cast<unsigned int>(*mismatch.second));
            }
            catch (...) {
                sycl::free(device_source, queue);
                sycl::free(device_destination, queue);
                throw;
            }
            sycl::free(device_source, queue);
            sycl::free(device_destination, queue);
        }

        std::vector<unsigned char> host_source(16, 1);
        std::vector<unsigned char> host_destination(16, 0);
        bool rejected = false;
        try {
            submit_staged(host_source.data(),
                          host_destination.data(),
                          host_source.size(),
                          queue,
                          identity);
        }
        catch (const std::exception&) {
            rejected = true;
        }
        CCL_THROW_IF_NOT(rejected, "host memory was accepted by NVSHMEM staging");
    }
    catch (const std::exception& error) {
        validation_error = error.what();
    }
    catch (...) {
        validation_error = "unknown local staging validation error";
    }

    const std::string validation_phase_error = agree_phase(bootstrap_kvs,
                                                           key_prefix + "_STAGING",
                                                           pe_rank,
                                                           pe_count,
                                                           validation_error,
                                                           "ready");
    CCL_THROW_IF_NOT(validation_phase_error.empty(),
                     "NVSHMEM staging validation failed: ",
                     validation_phase_error);

    std::lock_guard<std::mutex> lock(mutex);
    staging_validation_completed = true;
}

void runtime::finalize_locked() noexcept {
    if (cuda_device >= 0 && oneccl_nvshmem_cuda_set_device(cuda_device) != 0) {
        LOG_WARN("failed to select the NVSHMEM CUDA device during finalization: ",
                 oneccl_nvshmem_last_error());
    }
    if (last_completion) {
        try {
            last_completion->wait_and_throw();
        }
        catch (const std::exception& error) {
            LOG_WARN("failed to drain NVSHMEM staging work: ", error.what());
        }
        catch (...) {
            LOG_WARN("failed to drain NVSHMEM staging work with an unknown error");
        }
        last_completion.reset();
    }
    if (arena != nullptr) {
        oneccl_nvshmem_free(arena);
        arena = nullptr;
    }
    source_lane = nullptr;
    destination_lane = nullptr;
    metadata_region = nullptr;
    signal_region = nullptr;
    arena_bytes = 0;
    lane_bytes = 0;

    if (owns_initialization) {
        oneccl_nvshmem_hostlib_finalize();
        owns_initialization = false;
    }
    device.reset();
    context.reset();
    pe_rank = -1;
    pe_count = 0;
    cuda_device = -1;
    communicator_count = 0;
    bootstrap_kvs.reset();
    key_prefix.clear();
    staging_validation_completed = false;
    current_state = state::finalized;
}

} // namespace ccl::nvshmem
