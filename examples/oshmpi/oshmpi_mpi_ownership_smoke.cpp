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

#include <mpi.h>
#include <shmem.h>

#include <cstring>
#include <iostream>

namespace {

int fail(const char* message, int rank) {
    std::cerr << "Rank " << rank << ": " << message << std::endl;
    return 1;
}

void require_serialized_mpi(int rank) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Query_thread(&provided) != MPI_SUCCESS || provided < MPI_THREAD_SERIALIZED) {
        std::cerr << "Rank " << rank << ": OSHMPI lowered the MPI thread level" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int run_external_mpi(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided) != MPI_SUCCESS ||
        provided < MPI_THREAD_SERIALIZED) {
        return fail("MPI_THREAD_SERIALIZED is unavailable", -1);
    }

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int shmem_provided = SHMEM_THREAD_SINGLE;
    if (shmem_init_thread(SHMEM_THREAD_SERIALIZED, &shmem_provided) != SHMEM_SUCCESS ||
        shmem_provided < SHMEM_THREAD_SERIALIZED) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    require_serialized_mpi(rank);

    shmem_barrier_all();
    shmem_finalize();

    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized) {
        return fail("shmem_finalize finalized externally owned MPI", rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "PASS external-mpi" << std::endl;
    }
    MPI_Finalize();
    return 0;
}

int run_oshmpi_owned_mpi() {
    int provided = SHMEM_THREAD_SINGLE;
    if (shmem_init_thread(SHMEM_THREAD_SERIALIZED, &provided) != SHMEM_SUCCESS ||
        provided < SHMEM_THREAD_SERIALIZED) {
        return fail("SHMEM_THREAD_SERIALIZED is unavailable", -1);
    }

    const int rank = shmem_my_pe();
    require_serialized_mpi(rank);
    shmem_barrier_all();
    shmem_finalize();

    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
        return fail("OSHMPI did not finalize MPI that it initialized", rank);
    }
    if (rank == 0) {
        std::cout << "PASS oshmpi-owned-mpi" << std::endl;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (oshmpi_preserves_external_mpi() != 1) {
        return fail("patched OSHMPI ownership feature is unavailable", -1);
    }
    if (argc != 2) {
        return fail("usage: oshmpi_mpi_ownership_smoke <external|owned>", -1);
    }
    if (std::strcmp(argv[1], "external") == 0) {
        return run_external_mpi(argc, argv);
    }
    if (std::strcmp(argv[1], "owned") == 0) {
        return run_oshmpi_owned_mpi();
    }
    return fail("unknown ownership mode", -1);
}
