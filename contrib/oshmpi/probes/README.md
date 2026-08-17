# Probes

One-off programs written to answer a specific question about OSHMPI or the MPI
underneath it. They are not tests and nothing depends on them; they are kept
because `../DESIGN.md` cites their results, and a documented measurement nobody
can reproduce is worth very little.

Each probe is a `.cpp` plus the `.sbatch` that builds and runs it on Leonardo.
All of them expect `ONECCL_SOURCE_DIR` and source `../leonardo/env.sh`.

| probe | question it answered | finding |
|-------|----------------------|---------|
| `check_oshmpi_provenance.sh` | Which OSHMPI revision is this, and is it patched? | Pinned `ee5cf110`, OpenPA `0475704d`. Reports whether the ownership patch is applied. |
| `cuda_collectives.{cpp,sbatch}` | Do OSHMPI team collectives accept device pointers? | Data movement does (`fcollect`, `alltoall`); anything that computes or memcpys on the host does not (`broadcast`, `reduce`). See the results table in `../DESIGN.md`. |
| `mpi_device_allreduce.{cpp,sbatch}` | Is the device `allreduce` failure OSHMPI's fault or the MPI's? | The MPI's: with no accelerated collective component available, Open MPI reduces device operands on the host CPU. |
| `mpi_device_allreduce_matrix.sh` | — | Submits `mpi_device_allreduce.sbatch` across modes so one crash cannot hide the rest. |

The `mpi_device_allreduce` probe is the one still live: whether enabling UCC or
HCOLL lifts the device `allreduce` restriction has not been settled, and this is
the tooling to settle it.
