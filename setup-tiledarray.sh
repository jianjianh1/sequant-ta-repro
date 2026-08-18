#!/usr/bin/env bash
# Build TiledArray at commit cd53bd3 (the revision real MPQC tracks) with
# clang-21 + OpenBLAS, into third_party/tiledarray-cd53bd3-clang/install
# (the prefix CMakeLists.txt defaults to), then complete the install --
# this commit's install rules omit some MADNESS/TiledArray headers.
#
# Why cd53bd3 (not upstream tip, not the older 84411a6): its TA::einsum
# handles the multi-pair tensor-of-tensor tiles produced by this code's
# coarse occupied tiling (SPTC_COARSE_OCC); earlier commits crashed on them.
# NOTE: cd53bd3 lives ONLY on the jianjianh1/tiledarray fork (head of branch
# csv-cck-summa-root-fix), NOT on upstream ValeevGroup -- this is the exact
# revision real MPQC builds against, so the two sides share an identical TA.
#
# The repro executable MUST be built with the same compiler (clang++-21) --
# see README.md.
#
# Prereqs: clang-21 / clang++-21, cmake >= 3.21, an MPI (OpenMPI/MPICH),
# libopenblas-dev, libhwloc-dev, network access (fetches MADNESS + deps).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT=cd53bd3e04b28519b06a12b743c45824a588fff5
CLONE_DIR="$SCRIPT_DIR/third_party/tiledarray-cd53bd3-clang"
PREFIX="$CLONE_DIR/install"
BUILD="$CLONE_DIR/build"
JOBS="${JOBS:-$(nproc)}"

if [[ -d "$PREFIX/lib/cmake/tiledarray" ]]; then
  echo "TiledArray already installed at $PREFIX — remove it to force a rebuild."
  exit 0
fi

mkdir -p "$SCRIPT_DIR/third_party"
if [[ ! -d "$CLONE_DIR/.git" ]]; then
  git clone https://github.com/jianjianh1/tiledarray.git "$CLONE_DIR"
fi
git -C "$CLONE_DIR" fetch --depth 1 origin "$COMMIT" 2>/dev/null || git -C "$CLONE_DIR" fetch origin
git -C "$CLONE_DIR" checkout "$COMMIT"

cmake -S "$CLONE_DIR" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_CXX_COMPILER=clang++-21 -DCMAKE_C_COMPILER=clang-21 \
  -DCMAKE_CXX_FLAGS="-fno-lto -DLAPACK_FORTRAN_ADD_" \
  -DCMAKE_C_FLAGS="-fno-lto" \
  -DENABLE_MPI=ON \
  -DBLAS_PREFERENCE_LIST=OpenBLAS \
  -DTA_TTG=OFF
cmake --build "$BUILD" -j"$JOBS"
cmake --build "$BUILD" --target install

# Complete the install: cd53bd3's install rules omit all of madness/misc/,
# ~180 MADNESS headers, and TiledArray .ipp files. Copy the missing ones
# from the build/source trees (no-clobber preserves generated config headers).
cp -rn "$BUILD"/_deps/madness-src/src/madness/. "$PREFIX/include/madness/" 2>/dev/null || true
cp -rn "$CLONE_DIR"/src/TiledArray/. "$PREFIX/include/TiledArray/" 2>/dev/null || true

echo "TiledArray installed + completed at $PREFIX"
