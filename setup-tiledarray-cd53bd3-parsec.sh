#!/usr/bin/env bash
# Axis 1: build TiledArray cd53bd3 with clang-21 + the PaRSEC task backend
# (matching real MPQC's SIF: -DMADNESS_TASK_BACKEND=PaRSEC, clang-21,
# OpenBLAS; PaRSEC is pulled transitively by MADNESS's FetchContent — same
# MADNESS pin 666765ca6 MPQC uses; TA_TTG stays OFF). Reuses the existing
# cd53bd3 source checkout (out-of-source build into a separate prefix).
set -euo pipefail
REPO=/users/jianjian/sequant-ta-repro
SRC="$REPO/third_party/tiledarray-cd53bd3"
PREFIX="$REPO/third_party/tiledarray-cd53bd3-parsec"
BUILD="$PREFIX/build"
JOBS="$(nproc)"

[[ -d "$SRC/.git" ]] || { echo "missing source checkout $SRC"; exit 1; }
mkdir -p "$PREFIX"
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX/install" \
  -DCMAKE_CXX_COMPILER=clang++-21 -DCMAKE_C_COMPILER=clang-21 \
  -DCMAKE_CXX_FLAGS="-fno-lto -DLAPACK_FORTRAN_ADD_" \
  -DCMAKE_C_FLAGS="-fno-lto" \
  -DENABLE_MPI=ON \
  -DBLAS_PREFERENCE_LIST=OpenBLAS \
  -DMADNESS_TASK_BACKEND=PaRSEC \
  -DTA_TTG=OFF
cmake --build "$BUILD" -j"$JOBS"
cmake --build "$BUILD" --target install
# complete the install (cd53bd3 install rules omit many headers — see
# MPQC_COMPARISON.md §5); patch from this build's own sources.
cp -rn "$BUILD"/_deps/madness-src/src/madness/. "$PREFIX/install/include/madness/" 2>/dev/null || true
cp -rn "$SRC"/src/TiledArray/. "$PREFIX/install/include/TiledArray/" 2>/dev/null || true
echo "DONE_PARSEC: $PREFIX/install"
