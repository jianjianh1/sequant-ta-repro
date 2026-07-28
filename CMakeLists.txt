cmake_minimum_required(VERSION 3.21)
project(sequant_ta_repro CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Sequential LP64 BLAS: single-threaded, leaving the cores for TA's own
# task scheduler.
set(BLA_VENDOR Intel10_64lp_seq)
find_package(BLAS REQUIRED)

# Prepend the TiledArray install prefix so transitive find_dependency()
# calls (boost_headers, umpire, blaspp, range-v3, ...) resolve through TA's
# own lib/cmake/ subdirectories rather than expecting a system install.
# Build this prefix with setup-tiledarray.sh.
#
# Pinned to commit 84411a6 (NOT the newer upstream tip): confirmed a real
# internal bug in TA::einsum for a flat-operand x ToT-operand contraction
# that shares AND contracts an outer index (segfault / Boost bounds
# assertion) in the original checkout used before this pin, hit by ~1/3 of
# the real PNO-CCSD T1/T2 residual terms; 84411a6 computes the correct
# result for the identical pattern, verified against an independent numpy
# ground truth. Override TA_INSTALL_DIR to test a different commit.
set(TA_INSTALL_DIR "${CMAKE_SOURCE_DIR}/third_party/tiledarray-84411a6/install"
    CACHE PATH "TiledArray install prefix")
list(PREPEND CMAKE_PREFIX_PATH "${TA_INSTALL_DIR}")
find_package(tiledarray QUIET
    HINTS "${TA_INSTALL_DIR}/lib/cmake/tiledarray")

if(tiledarray_FOUND)
    message(STATUS "TiledArray found — building ta_sequant_native_residual_main")

    # Native SeQuant TiledArrayGenerator output (generated_t{1,2}_residual.cpp,
    # pasted verbatim -- see src/ta_sequant_native_residual.h for
    # provenance/regeneration instructions) driven against real leaf data via
    # the TATensors loader (src/ta_tensor_loader.h).
    add_executable(ta_sequant_native_residual_main
        src/ta_sequant_native_residual_main.cpp
        src/generated_t1_residual.cpp
        src/generated_t2_residual.cpp)
    target_link_libraries(ta_sequant_native_residual_main PRIVATE tiledarray)
    target_include_directories(ta_sequant_native_residual_main
        PRIVATE "${CMAKE_SOURCE_DIR}/src")

    # Warm-loop T2: persistent cache of the t-independent DF/CSV block
    # (MPQC's volatile/persistent split), SSA-split from the generated T2.
    if(EXISTS "${CMAKE_SOURCE_DIR}/gen_split/t2_split.cpp")
        add_executable(ta_warm_t2
            gen_split/ta_warm_t2_main.cpp
            gen_split/t2_split.cpp)
        target_link_libraries(ta_warm_t2 PRIVATE tiledarray)
        target_include_directories(ta_warm_t2
            PRIVATE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/gen_split")
    endif()
else()
    message(STATUS "TiledArray not found — run setup-tiledarray.sh first")
endif()
