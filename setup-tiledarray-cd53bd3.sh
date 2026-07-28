#!/usr/bin/env bash
# Phase 3: build the TiledArray revision that REAL MPQC (mpqc4) tracks
# (external/versions.cmake: MPQC_TRACKED_TILEDARRAY_TAG =
# cd53bd3e04b28519b06a12b743c45824a588fff5), OpenBLAS, into a throwaway
# prefix, to test whether it fixes the multi-pair-tile SparseShape crash
# that blocks the repro from matching MPQC's coarse ToT tiling on 84411a6.
set -euo pipefail
REPO=/users/jianjian/sequant-ta-repro
COMMIT=cd53bd3e04b28519b06a12b743c45824a588fff5
SHORT=cd53bd3
CLONE_DIR="$REPO/third_party/tiledarray-$SHORT"
JOBS="$(nproc)"

if [[ -d "$CLONE_DIR/install/lib/cmake/tiledarray" ]]; then
  echo "already installed at $CLONE_DIR/install"; exit 0
fi
mkdir -p "$REPO/third_party"
if [[ ! -d "$CLONE_DIR/.git" ]]; then
  git clone https://github.com/ValeevGroup/tiledarray.git "$CLONE_DIR"
fi
git -C "$CLONE_DIR" fetch --depth 1 origin "$COMMIT" 2>/dev/null || git -C "$CLONE_DIR" fetch origin
git -C "$CLONE_DIR" checkout "$COMMIT"
git -C "$CLONE_DIR" submodule update --init --recursive 2>/dev/null || true

cd "$CLONE_DIR"
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CLONE_DIR/install" \
  -DCMAKE_CXX_FLAGS="-fno-lto -DLAPACK_FORTRAN_ADD_" \
  -DCMAKE_C_FLAGS="-fno-lto" \
  -DENABLE_MPI=ON \
  -DBLAS_PREFERENCE_LIST=OpenBLAS
cmake --build build -j"$JOBS"
cmake --build build --target install
echo "DONE: TiledArray $SHORT installed to $CLONE_DIR/install"
