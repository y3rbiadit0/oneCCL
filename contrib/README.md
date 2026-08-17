# contrib

Supporting material that is not part of the oneCCL build: dependency patches,
design notes, and site-specific build and validation scripts.

Nothing here is compiled by the top-level `CMakeLists.txt`, and nothing here is
installed. This is deliberately not `examples/`, whose subdirectories are all
API sample programs built against an installed oneCCL.

- `oshmpi/` - the OSHMPI backend (`CCL_BACKEND=oshmpi`): the OSHMPI ownership
  patch the backend requires, and the scripts used to build and validate it on
  CINECA Leonardo.
