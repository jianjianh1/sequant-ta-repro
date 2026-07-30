#ifndef SPTC_AUX_K_BATCHING_H
#define SPTC_AUX_K_BATCHING_H

#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "ta_tensors.h"

// ---------------------------------------------------------------------------
// aux-Κ batching for the DF half-transform block (generated_t2_residual.cpp
// lines 487-491). The repro's port of MPQC's opt-in
// `make_batched_custom_evaluator` (sequant-fork eval.hpp:1129; wired in
// cck.ipp:1601-1645), documented as MPQC_EVALUATION.md stage S10.
//
// The block computes, unbatched:
//     I_ap2_μ̃_Κ[i,i,μ̃,Κ;a1] = einsum(g_μ̃_μ̃_Κ, C_ap2_μ̃)   // giant cell-bound
//     CSE37[i,i,Κ;a1,a3]     = einsum(I_ap2_μ̃_Κ, C_μ̃_ap2)
//     Itmp[i,i,Κ;a1,a4]      = einsum(CSE37, t_ap2_ap2_i_i)
//     I_i_i_ap2_ap2[i,i;a1,a2] += einsum(Itmp, CSE37)       // Κ contracted here
//
// Κ is contracted at the block root, so the block's contribution is a SUM over
// Κ. Streaming Κ in tile-aligned batches and summing the per-batch partials is
// byte-identical math (same cell count, same flops) but bounds peak memory:
// the giant μ̃Κ intermediate is only ever formed one Κ-batch wide. Mirrors MPQC:
// Κ sliced by aux_target_size (elements), partials summed, and the block-sparse
// screening threshold divided by n_batches so no block is screened away that
// the unbatched op keeps (cck.ipp:1620-1640).
//
// Memory bound, not a speed lever: the per-cell ToT-einsum cost is unchanged
// (MPQC_COMPARISON.md §11); its payoff is fitting the intermediate that
// otherwise OOMs at hexane scale.
// ---------------------------------------------------------------------------

// Build a copy of the flat DF tensor g[μ̃',μ̃,Κ] restricted to the Κ tiles
// [kt_lo, kt_hi), with the Κ axis rebased to start at 0. Done by copying tiles
// (norms computed from data, like build_sparse_array) rather than a TA
// `.block()` expression — a standalone block-slice of a sparse array deadlocks
// / trips "RMI thread not running" on this TA fork.
inline TA::TSpArrayD slice_g_over_K(const TA::TSpArrayD& g, std::size_t kt_lo,
                                    std::size_t kt_hi) {
  TA::World& world = g.world();
  const auto& gtr = g.trange();
  const TA::TiledRange1& ktr = gtr.dim(2);
  const std::size_t base = ktr.tile(kt_lo).first;  // element offset to subtract

  // Sub Κ TiledRange1, rebased to 0.
  std::vector<std::size_t> kb;
  kb.push_back(0);
  for (std::size_t kt = kt_lo; kt < kt_hi; ++kt)
    kb.push_back(ktr.tile(kt).second - base);
  TA::TiledRange1 ksub(kb.begin(), kb.end());
  TA::TiledRange sub_tr(
      {gtr.dim(0), gtr.dim(1), ksub});
  const auto& sub_tiles = sub_tr.tiles_range();
  const auto& g_tiles = gtr.tiles_range();

  // Pass 1: Frobenius norms of the copied (local, nonzero) tiles; global-sum so
  // every rank has the full norm tensor for the SparseShape.
  TA::Tensor<float> sub_norms(sub_tiles, 0.0f);
  const std::size_t n0 = gtr.dim(0).tile_extent();
  const std::size_t n1 = gtr.dim(1).tile_extent();
  for (std::size_t i0 = 0; i0 < n0; ++i0)
    for (std::size_t i1 = 0; i1 < n1; ++i1)
      for (std::size_t kt = kt_lo; kt < kt_hi; ++kt) {
        const std::size_t src_ord =
            g_tiles.ordinal(std::array<std::size_t, 3>{{i0, i1, kt}});
        if (g.is_zero(src_ord) || !g.is_local(src_ord)) continue;
        const auto& tile = g.find(src_ord).get();
        double ss = 0.0;
        for (std::size_t e = 0; e < tile.range().volume(); ++e)
          ss += tile.data()[e] * tile.data()[e];
        sub_norms(std::array<std::size_t, 3>{{i0, i1, kt - kt_lo}}) =
            static_cast<float>(std::sqrt(ss));
      }
  world.gop.sum(sub_norms.data(), sub_norms.size());

  TA::SparseShape<float> sub_shape(world, sub_norms, sub_tr);
  TA::TSpArrayD g_b(world, sub_tr, sub_shape);

  // Pass 3: copy each local nonzero g_b tile from the corresponding g tile
  // (fetched via find(); remote-safe), rebasing the tile's Range.
  for (auto it = g_b.begin(); it != g_b.end(); ++it) {
    const auto ord = it.ordinal();
    if (g_b.is_zero(ord) || !g_b.is_local(ord)) continue;
    auto idx = sub_tiles.idx(ord);  // {i0, i1, kt_sub}
    const std::size_t src_ord = g_tiles.ordinal(std::array<std::size_t, 3>{
        {(std::size_t)idx[0], (std::size_t)idx[1],
         (std::size_t)idx[2] + kt_lo}});
    const auto& src = g.find(src_ord).get();
    TA::Tensor<double> dst(sub_tr.make_tile_range(ord), 0.0);
    std::copy(src.data(), src.data() + src.range().volume(), dst.data());
    g_b.set(ord, std::move(dst));
  }
  world.gop.fence();
  return g_b;
}

// aux_target_size = target number of Κ ELEMENTS per batch (0 disables; matches
// MPQC's `batch:aux_target_size`). Accumulates into `acc` (= I_i_i_ap2_ap2,
// which already holds this residual's earlier contributions on entry).
inline void accumulate_df_halftransform_batched(
    ArrayToT& acc,                    // I_i_i_ap2_ap2 (accumulated into)
    const TA::TSpArrayD& g_μ̃_μ̃_Κ,     // g0  (μ̃', μ̃, Κ)
    const ArrayToT& C_ap2_μ̃,          // c2_tot
    const ArrayToT& C_μ̃_ap2,          // c2_tot
    const ArrayToT& t_ap2_ap2_i_i,    // t_i_i_a_a_tot
    std::size_t aux_target_size) {
  TA::World& world = g_μ̃_μ̃_Κ.world();
  const TA::TiledRange1& ktr = g_μ̃_μ̃_Κ.trange().dim(2);
  const bool dbg = std::getenv("SPTC_AUX_DEBUG") != nullptr;

  // Group Κ tiles into batches of ~aux_target_size elements (tile-aligned),
  // mirroring MPQC's mode_batches_of_trange1 (result.hpp:249).
  std::vector<std::pair<std::size_t, std::size_t>> batches;  // [tile_lo, tile_hi)
  {
    const std::size_t t_first = ktr.tiles_range().first;
    const std::size_t t_last = ktr.tiles_range().second;
    std::size_t tstart = t_first, elems = 0;
    for (std::size_t t = t_first; t < t_last; ++t) {
      const auto& trng = ktr.tile(t);
      elems += (trng.second - trng.first);
      if (elems >= aux_target_size || t + 1 == t_last) {
        batches.push_back({tstart, t + 1});
        tstart = t + 1;
        elems = 0;
      }
    }
  }
  const std::size_t n_batches = batches.size();
  if (dbg && world.rank() == 0)
    std::fprintf(stderr, "[auxk] n_batches=%zu Kext=%zu ntiles=%zu\n",
                 n_batches, (std::size_t)ktr.extent(),
                 (std::size_t)ktr.tile_extent());

  // Lower the block-sparse screening threshold for the batched region so a
  // block surviving in the full op is not screened in a narrower Κ-batch.
  const float thr0 = TA::SparseShape<float>::threshold();
  if (n_batches > 1)
    TA::SparseShape<float>::threshold(thr0 / static_cast<float>(n_batches));

  std::size_t bi = 0;
  for (const auto& b : batches) {
    if (dbg && world.rank() == 0)
      std::fprintf(stderr, "[auxk] batch %zu/%zu tiles[%zu,%zu)\n", bi,
                   n_batches, b.first, b.second);

    TA::TSpArrayD g_b = slice_g_over_K(g_μ̃_μ̃_Κ, b.first, b.second);
    if (dbg && world.rank() == 0) std::fprintf(stderr, "[auxk]  g_b sliced\n");

    // Verbatim transcription of generated_t2_residual.cpp:487-491 with g→g_b
    // and the root accumulating into `acc` (all intermediates Κ-batch wide).
    ArrayToT I_ap2_μ̃_Κ_b, CSE37_b, Itmp_b;
    I_ap2_μ̃_Κ_b("i_1,i_2,μ̃_19906,Κ_1;a_1") =
        TA::einsum(g_b("μ̃_19905,μ̃_19906,Κ_1"),
                   C_ap2_μ̃("i_1,i_2,μ̃_19905;a_1"),
                   "i_1,i_2,μ̃_19906,Κ_1;a_1")("i_1,i_2,μ̃_19906,Κ_1;a_1");
    CSE37_b("i_2,i_1,Κ_1;a_1,a_3") =
        TA::einsum(I_ap2_μ̃_Κ_b("i_1,i_2,μ̃_19906,Κ_1;a_1"),
                   C_μ̃_ap2("i_1,i_2,μ̃_19906;a_3"),
                   "i_2,i_1,Κ_1;a_1,a_3")("i_2,i_1,Κ_1;a_1,a_3");
    I_ap2_μ̃_Κ_b = ArrayToT();  // release the (batch-wide) giant intermediate
    Itmp_b("i_1,i_2,Κ_1;a_1,a_4") =
        TA::einsum(CSE37_b("i_2,i_1,Κ_1;a_1,a_3"),
                   t_ap2_ap2_i_i("i_1,i_2;a_3,a_4"),
                   "i_1,i_2,Κ_1;a_1,a_4")("i_1,i_2,Κ_1;a_1,a_4");
    acc("i_1,i_2;a_1,a_2") +=
        TA::einsum(Itmp_b("i_1,i_2,Κ_1;a_1,a_4"),
                   CSE37_b("i_2,i_1,Κ_1;a_2,a_4"),
                   "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2");
    world.gop.fence();  // free this batch's intermediates before the next
    if (dbg && world.rank() == 0)
      std::fprintf(stderr, "[auxk]  batch %zu accumulated\n", bi);
    ++bi;
  }

  if (n_batches > 1) TA::SparseShape<float>::threshold(thr0);
}

#endif  // SPTC_AUX_K_BATCHING_H
