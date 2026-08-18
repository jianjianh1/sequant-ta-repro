cmake_minimum_required(VERSION 3.21)
project(sequant_ta_repro CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TiledArray install prefix. Build it with ./setup-tiledarray.sh, which
# pins commit cd53bd3 -- the TiledArray revision real MPQC tracks -- whose
# TA::einsum correctly handles the multi-pair tensor-of-tensor tiles this
# code's coarse occupied tiling produces (older commits crashed on them).
# BLAS (OpenBLAS), MPI, Boost, umpire, range-v3, etc. all resolve
# transitively through this prefix's lib/cmake, so no separate find_package
# is needed. Build the repro with the SAME compiler as TA (clang++-21) --
# see README.md.
set(TA_INSTALL_DIR "${CMAKE_SOURCE_DIR}/third_party/tiledarray-cd53bd3-clang/install"
    CACHE PATH "TiledArray install prefix")
list(PREPEND CMAKE_PREFIX_PATH "${TA_INSTALL_DIR}")
find_package(tiledarray QUIET HINTS "${TA_INSTALL_DIR}/lib/cmake/tiledarray")

# Optional diagnostic / benchmark tools (see tools/README.md); off by
# default so the core build is just the residual driver.
option(SPTC_BUILD_TOOLS "Build the investigation tools under tools/" OFF)

if(tiledarray_FOUND)
    message(STATUS "TiledArray found — building ta_sequant_native_residual_main")

    # The explicit SeQuant -> TiledArray contraction sequence
    # (generated_t{1,2}_residual.cpp, pasted verbatim; see
    # src/ta_sequant_native_residual.h for provenance and regeneration),
    # driven against real leaf data via src/ta_tensor_loader.h.
    add_executable(ta_sequant_native_residual_main
        src/ta_sequant_native_residual_main.cpp
        src/generated_t1_residual.cpp
        src/generated_t2_residual.cpp)
    target_link_libraries(ta_sequant_native_residual_main PRIVATE tiledarray)
    target_include_directories(ta_sequant_native_residual_main
        PRIVATE "${CMAKE_SOURCE_DIR}/src")

    # aux-Κ-batching variant: identical driver but with the DF half-transform
    # block gated by SPTC_AUX_TARGET_SIZE (see src/aux_k_batching.h,
    # src/generated_t2_residual_auxbatch.cpp). SPTC_AUX_TARGET_SIZE=0 → the
    # original einsum block (a faithful A/B against the shipped binary);
    # >0 → Κ streamed in batches to bound the giant intermediate. Owning-ToT
    # (multi-rank-stable, required for the hexane run) is set per-target so all
    # its TUs agree on the ArrayToT inner-tile type.
    add_executable(ta_auxbatch_main
        src/ta_sequant_native_residual_main.cpp
        src/generated_t1_residual.cpp
        src/generated_t2_residual_auxbatch.cpp)
    target_link_libraries(ta_auxbatch_main PRIVATE tiledarray)
    target_include_directories(ta_auxbatch_main PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_compile_definitions(ta_auxbatch_main PRIVATE SPTC_OWNING_TOT)

    # One-process/one-product driver for the versioned MPQC-derived isolated
    # einsum suite in jianjianh1/mpqc-benchmark. Inputs are reconstructed before
    # timing; the only timed operation is one fence-bracketed TA::einsum.
    add_executable(ta_einsum_replay src/ta_einsum_replay_main.cpp)
    target_link_libraries(ta_einsum_replay PRIVATE tiledarray)
    target_include_directories(ta_einsum_replay PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_compile_definitions(ta_einsum_replay PRIVATE SPTC_OWNING_TOT)

    if(SPTC_BUILD_TOOLS AND EXISTS "${CMAKE_SOURCE_DIR}/tools/gen_split/t2_split.cpp")
        # Warm-loop T2 benchmark: persistent cache of the t-independent
        # DF/CSV block (MPQC's volatile/persistent split), SSA-split from
        # the generated T2. See tools/README.md.
        add_executable(ta_warm_t2
            tools/gen_split/ta_warm_t2_main.cpp
            tools/gen_split/t2_split.cpp)
        target_link_libraries(ta_warm_t2 PRIVATE tiledarray)
        target_include_directories(ta_warm_t2
            PRIVATE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/tools/gen_split")
    endif()

    if(SPTC_BUILD_TOOLS AND EXISTS "${CMAKE_SOURCE_DIR}/tools/gap_microbench.cpp")
        # Achievable-speedup ceiling for the cold CSE37 hotspot: TA::einsum
        # (L487-488) vs hand-written per-pair BLAS GEMM on identical data.
        # See tools/gap_microbench.cpp. Owning-ToT (single-node, thread-safe).
        add_executable(gap_microbench tools/gap_microbench.cpp)
        target_link_libraries(gap_microbench PRIVATE tiledarray)
        target_include_directories(gap_microbench
            PRIVATE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/tools")
        target_compile_definitions(gap_microbench PRIVATE SPTC_OWNING_TOT)
    endif()

    # Runtime-evaluator variant: computes the T2 residual via sequant::evaluate()
    # over the L2 forest (MPQC's runtime path) instead of the static generated
    # einsum sequence — the non-canonical runtime-evaluator control. Gated
    # (pulls the SeQuant lib). Owning-ToT is REQUIRED: the evaluator's
    # cross-thread lazy cleanup dangles arena inner cells.
    option(SPTC_BUILD_RUNTIME_EVAL "Build the sequant::evaluate() runtime T2 driver" OFF)
    if(SPTC_BUILD_RUNTIME_EVAL)
        # Put the SeQuant install prefix + its FetchContent dep build dirs on the
        # search path so its transitive deps (bundled boost_* configs, Eigen3,
        # range-v3, libperm, polymorphic_variant, TiledArray) resolve. Eigen3
        # must be found FIRST so the Eigen3::Eigen target exists when SeQuant's
        # config references it.
        set(_SQ_INSTALL "/users/jianjian/sequant-fork/install-eval")
        set(_SQ_DEPS "/users/jianjian/sequant-fork/build-eval/_deps")
        list(PREPEND CMAKE_PREFIX_PATH
            "${_SQ_INSTALL}"
            "${_SQ_DEPS}/eigen3-build" "${_SQ_DEPS}/range-v3-build"
            "${_SQ_DEPS}/libperm-build" "${_SQ_DEPS}/polymorphic_variant-build")
        find_package(Eigen3 CONFIG QUIET)
        find_package(SeQuant CONFIG QUIET
            HINTS "${_SQ_INSTALL}/lib/cmake/sequant")
        if(SeQuant_FOUND)
            # Quarantined experiment (docs/HARNESS_VS_EXPERIMENTS.md): the
            # sequant::evaluate() runtime-evaluator control, moved out of src/
            # (the harness) into experiments/. Default-off; not the benchmark.
            add_executable(ta_runtime_eval_main experiments/ta_runtime_eval_main.cpp)
            target_link_libraries(ta_runtime_eval_main PRIVATE
                tiledarray SeQuant::SeQuant)
            target_include_directories(ta_runtime_eval_main
                PRIVATE "${CMAKE_SOURCE_DIR}/src")
            target_compile_definitions(ta_runtime_eval_main PRIVATE SPTC_OWNING_TOT)
        else()
            message(STATUS "SPTC_BUILD_RUNTIME_EVAL=ON but SeQuant not found")
        endif()
    endif()
else()
    message(STATUS "TiledArray not found — run ./setup-tiledarray.sh first")
endif()
