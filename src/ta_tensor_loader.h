#ifndef SPTC_TA_TENSOR_LOADER_H
#define SPTC_TA_TENSOR_LOADER_H

// Shared TATensors-loading logic: loads all leaf tensors from a trace
// directory as TiledArray sparse arrays (flat + tensor-of-tensor), applying
// the tiling chosen in ta_builder.h. Used by the residual driver.

#include <tiledarray.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

#include <TiledArray/conversions/foreach.h>
#include <TiledArray/tensor/arena_kernels.h>

#include "coo_loader.h"
#include "ta_builder.h"
#include "ta_stage.h"
#include "ta_tensors.h"

namespace fs = std::filesystem;

/// Port of MPQC's compact_csv_coeffs (mpqc4 mbpt/csv.ipp:44-64): coalesce each
/// ToT coefficient tile into ONE contiguous, single-page, constant-stride arena
/// slab. That is the layout the strided-DGEMM fast path (arena_einsum.h
/// arena_strided_dgemm_ce_e/ce_ce) requires to issue one BLAS call per k-run
/// instead of a per-inner-cell fallback. Incrementally-built (uncompacted) ToT
/// tiles span multiple arena pages with non-constant inter-cell stride and so
/// never take the fast path -- this is the single-thread lever MPQC has and the
/// repro lacked. No-op unless the inner cells are arena views (i.e. the default
/// arena build; compiled out for owning ToT). Runtime-gated by
/// SPTC_COMPACT_COEFFS=1 so one binary can A/B compaction on/off.
inline void compact_tot_coeffs(ArrayToT& a) {
  using Tile = typename ArrayToT::value_type;
  using Inner = typename Tile::value_type;
  if constexpr (TA::is_tensor_view_v<Inner>) {
    if (!a.is_initialized()) return;
    TA::foreach_inplace(a, [](Tile& tile) {
      tile = TA::detail::arena_compact<Tile>(tile);
      return tile.norm();
    });
  }
}

// Emit one synthetic LOAD row to mark per-(mol, np) tensor-load cost. Lives
// outside the trial/equation loops so per-eq aggregators ignore it via the
// equation == "_LOAD_" sentinel. peak_rss_kb reflects the post-load high
// water mark, which dominates the rest of the run for small equations.
inline void print_load_row(TA::World& world, const std::string& mol, double load_s) {
  if (world.rank() != 0) return;
  std::ios old_state(nullptr);
  old_state.copyfmt(std::cout);
  std::cout << std::setprecision(17);
  std::cout << mol << ",_LOAD_,load,0," << world.size() << "," << load_s
            << ",0,0," << get_peak_rss_kb() << ",0,0,0,0,\n";
  std::cout.copyfmt(old_state);
  std::cout << std::flush;
}

/// Load one tensor, deriving its tile sizes from its own real shape (see
/// adaptive_tile_sizes() in ta_builder.h).
///
/// SPTC_FLAT_TILES_PER_DIM (2026-07-27), if set, overrides the target
/// tile count for FLAT arrays only, independent of SPTC_TILES_PER_DIM
/// (which otherwise also governs ToT arrays' non-pair-key outer dims via
/// build_tot_array() below) -- flat and ToT tiling may not share the same
/// optimum now that both have been re-tuned once already (see
/// adaptive_tile_sizes()'s own comment on knob interaction).
inline TA::TSpArrayD load_one(TA::World& world, const std::string& path,
                              const std::string& label) {
  auto coo = load_coo(path);
  std::vector<std::size_t> tile_sizes;
  if (const char* v = std::getenv("SPTC_FLAT_TILES_PER_DIM")) {
    std::size_t n = static_cast<std::size_t>(std::atoi(v));
    if (n > 0) {
      tile_sizes = adaptive_tile_sizes(coo.shape, coo.rank, n,
                                       /*check_shared_env=*/false);
    }
  }
  if (tile_sizes.empty())
    tile_sizes = adaptive_tile_sizes(coo.shape, coo.rank);
  return build_sparse_array(world, coo, tile_sizes, label);
}

/// Load one PNO/OSV-restricted tensor as a genuine tensor-of-tensor array.
/// outer_rank/inner_rank/pair_key_rank per the table in ta_tensors.h.
inline ArrayToT load_one_tot(TA::World& world, const std::string& path,
                             int outer_rank, int inner_rank, int pair_key_rank,
                             const std::string& label) {
  auto coo = load_coo(path);
  return build_tot_array<ArrayToT>(world, coo, outer_rank, inner_rank,
                                   pair_key_rank, label);
}

/// Load all 11 tensors for a molecule as TiledArray sparse arrays, plus
/// tensor-of-tensor duplicates of the 4 PNO/OSV-restricted leaves.
inline TATensors load_ta_tensors(TA::World& world, const std::string& dir) {
  TATensors ts;
  auto t0 = std::chrono::high_resolution_clock::now();

  if (world.rank() == 0)
    std::cerr << "Loading tensors from " << dir << " ...\n";

  ts.g0 = load_one(world, dir + "/g_m_1_m_2_\xCE\x9A_1.txt", "g0");
  ts.g1 = load_one(world, dir + "/g_i_1_i_2_\xCE\x9A_1.txt", "g1");
  ts.g = load_one(world, dir + "/g_i_1_m_1_\xCE\x9A_1.txt", "g");
  ts.c1 = load_one(world, dir + "/C_m_1_a_1_i_1.txt", "c1");
  ts.c2 = load_one(world, dir + "/C_m_1_a_1_i_1_i_2.txt", "c2");
  ts.f_i_i = load_one(world, dir + "/f_i_1_i_2.txt", "f_i_i");
  ts.f_i_m = load_one(world, dir + "/f_i_1_m_1.txt", "f_i_m");
  ts.f_m_m = load_one(world, dir + "/f_m_1_m_2.txt", "f_m_m");
  ts.s_m_m = load_one(world, dir + "/s_m_1_m_2.txt", "s_m_m");
  ts.t_i_a = load_one(world, dir + "/t_i_1_a_1.txt", "t_i_a");
  ts.t_i_i_a_a = load_one(world, dir + "/t_i_1_i_2_a_1_a_2.txt", "t_i_i_a_a");

  // outer_rank, inner_rank, pair_key_rank per the table in ta_tensors.h:
  // c1 outer=(i,m) inner=(a) pair=(i); c2 outer=(i,i,m) inner=(a) pair=(i,i);
  // t_i_a outer=(i) inner=(a) pair=(i); t_i_i_a_a outer=(i,i) inner=(a,a) pair=(i,i).
  ts.c1_tot = load_one_tot(world, dir + "/C_m_1_a_1_i_1.txt", 2, 1, 1, "c1_tot");
  ts.c2_tot = load_one_tot(world, dir + "/C_m_1_a_1_i_1_i_2.txt", 3, 1, 2, "c2_tot");
  ts.t_i_a_tot = load_one_tot(world, dir + "/t_i_1_a_1.txt", 1, 1, 1, "t_i_a_tot");
  ts.t_i_i_a_a_tot =
      load_one_tot(world, dir + "/t_i_1_i_2_a_1_a_2.txt", 2, 2, 2, "t_i_i_a_a_tot");

  // MPQC-parity: compact the ToT coefficient/amplitude arrays to single-page
  // constant-stride so the strided-DGEMM fast path fires (see compact_tot_coeffs
  // above). Gated by SPTC_COMPACT_COEFFS=1; no-op for owning ToT.
  if (const char* v = std::getenv("SPTC_COMPACT_COEFFS"); v && std::atoi(v)) {
    compact_tot_coeffs(ts.c1_tot);
    compact_tot_coeffs(ts.c2_tot);
    compact_tot_coeffs(ts.t_i_a_tot);
    compact_tot_coeffs(ts.t_i_i_a_a_tot);
    if (world.rank() == 0)
      std::cerr << "  [compact] ToT coefficient arrays compacted "
                   "(SPTC_COMPACT_COEFFS=1)\n";
  }

  world.gop.fence();
  auto t1 = std::chrono::high_resolution_clock::now();
  double load_s = std::chrono::duration<double>(t1 - t0).count();
  if (world.rank() == 0)
    std::cerr << "  Load time: " << load_s << " s\n";
  std::string mol = fs::path(dir).filename().string();
  print_load_row(world, mol, load_s);

  return ts;
}

#endif  // SPTC_TA_TENSOR_LOADER_H
