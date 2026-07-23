#ifndef SPTC_TA_TENSOR_LOADER_H
#define SPTC_TA_TENSOR_LOADER_H

// Shared TATensors-loading logic, factored out of ta_benchmark_main.cpp so
// ta_trace_benchmark_main.cpp (mpqc_trace_equations.h's checksum-validation
// harness) can reuse it without duplicating the COO-load/tiling path.

#include <tiledarray.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

#include "coo_loader.h"
#include "ta_builder.h"
#include "ta_stage.h"
#include "ta_tensors.h"

namespace fs = std::filesystem;

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
inline TA::TSpArrayD load_one(TA::World& world, const std::string& path,
                              const std::string& label) {
  auto coo = load_coo(path);
  auto tile_sizes = adaptive_tile_sizes(coo.shape, coo.rank);
  return build_sparse_array(world, coo, tile_sizes, label);
}

/// Load one PNO/OSV-restricted tensor as a genuine tensor-of-tensor array.
/// outer_rank/inner_rank/pair_key_rank per the table in ta_tensors.h.
inline ArrayToT load_one_tot(TA::World& world, const std::string& path,
                             int outer_rank, int inner_rank, int pair_key_rank,
                             const std::string& label,
                             const std::string& real_tiling_sidecar = "") {
  auto coo = load_coo(path);
  return build_tot_array<ArrayToT>(world, coo, outer_rank, inner_rank,
                                   pair_key_rank, label, real_tiling_sidecar);
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
  std::string c2_tot_sidecar;
  if (const char* v = std::getenv("SPTC_REAL_TOT_TILING"); v && std::atoi(v) != 0) {
    std::string spec_dir = "ta-bench/data/tiling_specs";
    if (const char* d = std::getenv("SPTC_TILING_SPEC_DIR")) spec_dir = d;
    std::string mol = fs::path(dir).filename().string();
    c2_tot_sidecar = spec_dir + "/" + mol + "_c2_tot.tiling.txt";
    if (world.rank() == 0)
      std::cerr << "  [SPTC_REAL_TOT_TILING=1] c2_tot using real tiling from "
                << c2_tot_sidecar << "\n";
  }
  ts.c2_tot = load_one_tot(world, dir + "/C_m_1_a_1_i_1_i_2.txt", 3, 1, 2,
                           "c2_tot", c2_tot_sidecar);
  ts.t_i_a_tot = load_one_tot(world, dir + "/t_i_1_a_1.txt", 1, 1, 1, "t_i_a_tot");
  ts.t_i_i_a_a_tot =
      load_one_tot(world, dir + "/t_i_1_i_2_a_1_a_2.txt", 2, 2, 2, "t_i_i_a_a_tot");

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
