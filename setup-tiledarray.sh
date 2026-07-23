#!/usr/bin/env bash
# setup-tiledarray.sh — clone, build, and install TiledArray at the commit
# this repo's CMakeLists.txt is pinned to (84411a6), into
# third_party/tiledarray-84411a6/install (CMakeLists.txt's default
# TA_INSTALL_DIR).
#
# Why 84411a6 and not upstream tip: a real internal bug in TA::einsum for a
# flat-operand x ToT-operand contraction that shares AND contracts an outer
# index (segfault / Boost bounds assertion) was found in an earlier
# checkout, hit by ~1/3 of the real PNO-CCSD T1/T2 residual terms. 84411a6
# computes the correct result for the identical pattern (verified against
# an independent numpy ground truth). See CMakeLists.txt's own comment.
#
# Usage: ./setup-tiledarray.sh [--jobs N]
#
# Prerequisites: g++ >= 11 (C++20), cmake >= 3.21, MPI (OpenMPI or MPICH),
# OpenBLAS dev headers.
#   Debian/Ubuntu: sudo apt install build-essential g++-13 cmake libopenblas-dev libopenmpi-dev
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="$(nproc)"
COMMIT="84411a6"
CLONE_DIR="$SCRIPT_DIR/third_party/tiledarray-84411a6"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--jobs N]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -d "$CLONE_DIR/install/lib/cmake/tiledarray" ]]; then
  echo "TiledArray already installed at $CLONE_DIR/install — remove it to force a rebuild."
  exit 0
fi

mkdir -p "$SCRIPT_DIR/third_party"
if [[ ! -d "$CLONE_DIR" ]]; then
  git clone https://github.com/ValeevGroup/tiledarray.git "$CLONE_DIR"
fi
git -C "$CLONE_DIR" fetch --depth 1 origin "$COMMIT" 2>/dev/null || true
git -C "$CLONE_DIR" checkout "$COMMIT"

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

echo "TiledArray installed to $CLONE_DIR/install"
