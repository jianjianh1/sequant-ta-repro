#ifndef SPTC_TA_BUILDER_H
#define SPTC_TA_BUILDER_H

#include <tiledarray.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "coo_loader.h"

/// Derive per-dimension tile sizes from a tensor's REAL loaded shape,
/// targeting a fixed tile count per dimension. Replaces the prior
/// hardcoded-role tiling (occ=4, uocc=50, ri=200 regardless of the actual
/// molecule or tensor) — confirmed root cause of the OOM/abort failures in
/// the previous 47-equation sweep (commit 2eb61b2): those constants also
/// conflated genuinely different-sized dimensions (e.g. the ~114-wide PAO
/// axis and the ~600-wide per-pair PNO/virtual axis both used "uocc").
/// Sizing from each tensor's own shape avoids both the conflation and the
/// molecule-size mismatch.
inline std::vector<std::size_t> adaptive_tile_sizes(
    const std::array<std::size_t, 4>& shape, int rank,
    std::size_t target_tiles_per_dim = 8, bool check_shared_env = true) {
  // EXPERIMENT (2026-07-20, performance-parity investigation): allow
  // overriding the target tile count per dimension without touching the
  // shared default (which every ta-bench tool relies on) -- used to test
  // whether fewer/larger tiles reduce TA's per-task scheduling overhead
  // for the native SeQuant generator's ~500-statement whole-residual
  // computation. Not wired into any other tool's behavior unless the env
  // var is explicitly set.
  //
  // UPDATE (2026-07-26): the default of 8 was the confirmed optimum ONLY
  // relative to an UNPINNED process (the original sweep predates CPU
  // affinity pinning). Once the process is pinned to physical cores
  // (`taskset -c <physical-core-list>`) AND MAD_NUM_THREADS matches the
  // physical core count, the optimum SHIFTS COARSER --
  // SPTC_TILES_PER_DIM=6 then beats every value from 5 to 16 (~24-37%
  // faster than the old default=8 at the same pinned config); 4 still
  // times out (dense-intermediate-blowup risk on pair-key-adjacent tiling
  // still applies at that granularity). This mirrors MAD_NUM_THREADS's
  // own optimum flipping after pinning -- these knobs interact, so
  // re-sweep this one too after any further scheduling-level change
  // rather than assuming today's optimum is stable.
  //
  // `check_shared_env=false` lets a caller (e.g. load_one() below, via
  // SPTC_FLAT_TILES_PER_DIM) supply an already-resolved target that
  // bypasses this shared env lookup -- otherwise SPTC_TILES_PER_DIM,
  // when set, would silently override ANY explicit target_tiles_per_dim
  // argument, making a per-call-site override impossible.
  if (check_shared_env) {
    if (const char* v = std::getenv("SPTC_TILES_PER_DIM")) {
      std::size_t n = static_cast<std::size_t>(std::atoi(v));
      if (n > 0) target_tiles_per_dim = n;
    }
  }
  // Coarse-occ override (2026-07-27, performance-parity gap-closer,
  // env-gated): match real MPQC's occ_tile_size by tiling any dimension
  // whose extent equals the occupied-space size (SPTC_COARSE_OCC, e.g. 9
  // for ethane frozen-core) at SPTC_OCC_TILE (default 4) instead of the
  // adaptive size. Applied HERE so every tensor's occ-indexed dims — flat
  // arrays (via load_one) and ToT non-pair-key dims — share one identical
  // TiledRange1; build_tot_array applies the same override to pair-key
  // dims with a padded-uniform inner per tile. Consistency across all
  // occ-indexed dims is required: a contraction over occ needs matching
  // tilings on both operands (mismatch corrupts the heap — see §7).
  std::size_t occ_ext = 0, occ_tile = 4;
  if (const char* v = std::getenv("SPTC_COARSE_OCC")) {
    long n = std::atol(v); if (n > 0) occ_ext = static_cast<std::size_t>(n);
  }
  if (const char* v = std::getenv("SPTC_OCC_TILE")) {
    long n = std::atol(v); if (n > 0) occ_tile = static_cast<std::size_t>(n);
  }
  std::vector<std::size_t> sizes(rank);
  for (int d = 0; d < rank; ++d) {
    std::size_t extent = shape[d];
    if (occ_ext && extent == occ_ext)
      sizes[d] = std::min(extent, occ_tile);
    else
      sizes[d] = std::max<std::size_t>(1, extent / target_tiles_per_dim);
  }
  return sizes;
}

/// Diagnostic (env-gated SPTC_DUMP_PMAP, 2026-07-28): dump how this array's
/// nonzero tiles are distributed across MPI ranks. Tests the cold-gap
/// hypothesis in docs/MPQC_EVALUATION.md §8: MPQC's DF-carrying ToT arrays
/// inherit the CSV solver's pmap, while the repro takes TA's DEFAULT blocked
/// pmap here -- if the giant dense DF operand lands on only a few ranks, the
/// SUMMA of the giant intermediate is load-imbalanced regardless of tile size
/// (which the occ-tiling sweep already ruled out as the lever). Per-rank tile
/// counts are pmap-only, so they are identical whether the N ranks sit on N
/// nodes or one -- run `mpirun -np 16` on a single node to measure. No effect
/// unless SPTC_DUMP_PMAP is set; no math change.
template <typename Array>
inline void dump_pmap_distribution(TA::World& world, const Array& array,
                                   const std::string& label) {
  if (!std::getenv("SPTC_DUMP_PMAP") || label.empty()) return;
  const std::size_t nranks = world.size();
  std::vector<long> per_rank(nranks, 0);
  long local = 0;
  for (auto it = array.begin(); it != array.end(); ++it) ++local;
  per_rank[world.rank()] = local;
  world.gop.sum(per_rank.data(), per_rank.size());
  if (world.rank() == 0) {
    const std::size_t total = array.trange().tiles_range().volume();
    long occupied = 0, mx = 0, mn = -1, empty_ranks = 0;
    for (long c : per_rank) {
      occupied += c;
      mx = std::max(mx, c);
      if (mn < 0 || c < mn) mn = c;
      if (c == 0) ++empty_ranks;
    }
    std::cerr << "  PMAP " << label << ": " << occupied << " nonzero tiles / "
              << total << " total, over " << nranks << " ranks; per-rank min/max="
              << mn << "/" << mx << ", " << empty_ranks << " idle ranks [";
    for (std::size_t r = 0; r < nranks; ++r) {
      if (r) std::cerr << ",";
      std::cerr << per_rank[r];
    }
    std::cerr << "]\n";
  }
}

/// Build a TiledRange1 with uniform tile sizes for a dimension of given extent.
inline TA::TiledRange1 make_tr1(std::size_t extent, std::size_t tile_size) {
  std::vector<std::size_t> boundaries;
  for (std::size_t i = 0; i <= extent; i += tile_size)
    boundaries.push_back(i);
  if (boundaries.back() != extent) boundaries.push_back(extent);
  return TA::TiledRange1(boundaries.begin(), boundaries.end());
}

/// Build a TiledRange from per-dimension extents and tile sizes.
inline TA::TiledRange make_trange(const std::vector<std::size_t>& shape,
                                  const std::vector<std::size_t>& tile_sizes) {
  std::vector<TA::TiledRange1> tr1s;
  for (std::size_t d = 0; d < shape.size(); ++d)
    tr1s.push_back(make_tr1(shape[d], tile_sizes[d]));
  return TA::TiledRange(tr1s.begin(), tr1s.end());
}

/// Convert a COOTensor into a distributed TiledArray sparse array.
///
/// Three-pass algorithm:
///   Pass 1: Scan all entries, compute tile norms (all ranks, for SparseShape).
///   Pass 2: Create sparse shape and array (determines tile ownership).
///   Pass 3: Re-scan entries, only group entries for LOCAL tiles, set tiles.
inline TA::TSpArrayD build_sparse_array(TA::World& world,
                                        const COOTensor& coo,
                                        const std::vector<std::size_t>& tile_sizes,
                                        const std::string& label = "") {
  std::vector<std::size_t> shape(coo.shape.begin(),
                                 coo.shape.begin() + coo.rank);
  TA::TiledRange trange = make_trange(shape, tile_sizes);
  const auto& tiles_range = trange.tiles_range();

  // Pass 1: compute tile norms (all ranks need this for SparseShape)
  TA::Tensor<float> tile_norms(tiles_range, 0.0f);

  for (std::size_t i = 0; i < coo.values.size(); ++i) {
    std::vector<long> elem_idx(coo.rank);
    for (int d = 0; d < coo.rank; ++d)
      elem_idx[d] = static_cast<long>(coo.indices[i][d]);

    auto tidx = trange.element_to_tile(elem_idx);
    float val = static_cast<float>(coo.values[i]);
    tile_norms[tidx] += val * val;
  }
  tile_norms.inplace_unary([](float& x) { x = std::sqrt(x); });

  // Pass 2: create sparse shape and array (determines tile-to-rank mapping)
  TA::SparseShape<float> sp_shape(world, tile_norms, trange);
  TA::TSpArrayD array(world, trange, sp_shape);

  // Pass 3: re-scan entries, only collect entries for local non-zero tiles
  using ElemEntry = std::pair<std::vector<long>, double>;
  std::map<std::size_t, std::vector<ElemEntry>> tile_entries;

  for (std::size_t i = 0; i < coo.values.size(); ++i) {
    std::vector<long> elem_idx(coo.rank);
    for (int d = 0; d < coo.rank; ++d)
      elem_idx[d] = static_cast<long>(coo.indices[i][d]);

    auto tidx = trange.element_to_tile(elem_idx);
    auto ord = tiles_range.ordinal(tidx);

    if (array.is_zero(ord) || !array.is_local(ord)) continue;
    tile_entries[ord].emplace_back(std::move(elem_idx), coo.values[i]);
  }

  // Set each local tile
  for (auto& [ord, entries] : tile_entries) {
    auto tile_range = trange.make_tile_range(ord);
    TA::Tensor<double> tile(tile_range, 0.0);
    for (auto& [idx, val] : entries)
      tile[idx] = val;
    array.set(ord, std::move(tile));
  }

  // Set remaining local non-zero tiles (no COO entries) to zero
  for (auto it = array.begin(); it != array.end(); ++it) {
    auto ord = it.ordinal();
    if (!array.is_zero(ord) && array.is_local(ord) &&
        tile_entries.find(ord) == tile_entries.end()) {
      array.set(ord, TA::Tensor<double>(trange.make_tile_range(ord), 0.0));
    }
  }

  world.gop.fence();

  if (world.rank() == 0 && !label.empty()) {
    std::cerr << "  " << label << ": shape (";
    for (int d = 0; d < coo.rank; ++d) {
      if (d) std::cerr << ",";
      std::cerr << shape[d];
    }
    std::cerr << "), nnz=" << coo.values.size()
              << ", sparsity=" << (sp_shape.sparsity() * 100.0) << "%"
              << ", tiles=" << tiles_range.volume() << "\n";
  }
  dump_pmap_distribution(world, array, label);

  return array;
}

/// Build a tensor-of-tensor (ToT) array from COO data, reconstructing the
/// per-(occupied-index-tuple)-restricted PNO/OSV structure: `outer_rank`
/// leading dims are block-sparse/outer (e.g. i,m for C1); the remaining
/// `inner_rank` trailing dims are the per-pair-restricted virtual/PNO axes.
/// `pair_key_rank` (<= outer_rank) is how many LEADING outer dims form the
/// occupied-index tuple that determines a pair's PNO count (1 for C1/T1, 2
/// for C2/T2).
///
/// TILING CONTROL (the point of this repo): by default the pair-key
/// (occupied) dims are tiled singly — one pair per outer tile, each with
/// its own inner Range. Set `SPTC_COARSE_OCC=<occ_extent>` to instead tile
/// those dims coarsely at `SPTC_OCC_TILE` (default 4), matching real MPQC's
/// `occ_tile_size` (fewer, larger tiles → less task-scheduling overhead).
/// A coarse tile spans several pairs; `SPTC_COARSE_PAD=1` (default) gives
/// every position in the tile one padded-uniform inner size (max PNO count
/// over its pairs), while `SPTC_COARSE_PAD=0` keeps a ragged per-pair inner
/// size (matches MPQC's layout, no zero-padding). Coarse ToT tiling
/// requires a TiledArray whose einsum handles multi-pair tiles (>= cd53bd3);
/// older commits crash on them.
template <typename ArrayToT>
inline ArrayToT build_tot_array(TA::World& world, const COOTensor& coo,
                                int outer_rank, int inner_rank,
                                int pair_key_rank,
                                const std::string& label = "") {
  using InnerT = typename ArrayToT::value_type::value_type;  // TA::Tensor<double>

  // Coarse-occ tiling control (env-gated via SPTC_COARSE_OCC): tile the
  // pair-key (occupied) dims at SPTC_OCC_TILE like real MPQC instead of
  // forcing size 1, computing the padded-uniform per-tile inner size in
  // code (derived from this array's own pair_range so every ToT array
  // coarsens consistently). tile_pad_volume: pair-key tile multi-index ->
  // padded inner size, populated below when SPTC_COARSE_PAD=1.
  std::map<std::vector<long>, std::size_t> tile_pad_volume;
  std::size_t coarse_occ_ext = 0, coarse_occ_tile = 4;
  if (const char* v = std::getenv("SPTC_COARSE_OCC")) {
    long n = std::atol(v); if (n > 0) coarse_occ_ext = static_cast<std::size_t>(n);
  }
  if (const char* v = std::getenv("SPTC_OCC_TILE")) {
    long n = std::atol(v); if (n > 0) coarse_occ_tile = static_cast<std::size_t>(n);
  }
  const bool use_coarse = coarse_occ_ext > 0;
  // SPTC_COARSE_PAD=0 opts a coarse (multi-pair) outer tile into a RAGGED
  // per-pair inner size (each outer element keeps its own PNO count) rather
  // than one padded-uniform size per tile — exactly how real MPQC sizes its
  // ToT inner cells. This is the "per-position varying inner Range within
  // one tile" pattern the size-1 default was built to avoid (it crashed
  // TA::einsum on the older 84411a6 commit, eq10 SIGSEGV). The pinned
  // cd53bd3 commit handles it correctly (checksums match), so ragged
  // inner sizing removes the padding waste and matches MPQC exactly.
  bool coarse_pad = true;
  if (const char* v = std::getenv("SPTC_COARSE_PAD")) coarse_pad = std::atoi(v) != 0;
  const bool use_padded = use_coarse && coarse_pad;

  // Pass 0: group by pair key -> (min_inner, max_inner) across all inner
  // columns combined (C2/T2's two virtual axes share one per-pair domain).
  std::map<std::vector<long>, std::pair<long, long>> pair_range;
  // outer coordinate -> [(inner local coord, value), ...]
  std::map<std::vector<long>, std::vector<std::pair<std::vector<long>, double>>>
      data_by_outer;

  for (std::size_t i = 0; i < coo.values.size(); ++i) {
    std::vector<long> outer_coord(outer_rank), pair_key(pair_key_rank);
    for (int d = 0; d < outer_rank; ++d)
      outer_coord[d] = static_cast<long>(coo.indices[i][d]);
    for (int d = 0; d < pair_key_rank; ++d) pair_key[d] = outer_coord[d];

    std::vector<long> inner_coord_global(inner_rank);
    for (int d = 0; d < inner_rank; ++d)
      inner_coord_global[d] = static_cast<long>(coo.indices[i][outer_rank + d]);

    auto pr_it = pair_range.find(pair_key);
    if (pr_it == pair_range.end()) {
      // First row seen for this pair key: seed min/max from its own values.
      long lo = *std::min_element(inner_coord_global.begin(), inner_coord_global.end());
      long hi = *std::max_element(inner_coord_global.begin(), inner_coord_global.end());
      pair_range.emplace(pair_key, std::make_pair(lo, hi));
    } else {
      for (long v : inner_coord_global) {
        pr_it->second.first = std::min(pr_it->second.first, v);
        pr_it->second.second = std::max(pr_it->second.second, v);
      }
    }
    data_by_outer[outer_coord].emplace_back(std::move(inner_coord_global), coo.values[i]);
  }
  // Second pass to re-express inner coords as pair-local (needs pair_range
  // fully populated first, since a row's own min/max isn't necessarily the
  // pair's overall min/max).
  for (auto& [outer_coord, entries] : data_by_outer) {
    std::vector<long> pair_key(pair_key_rank);
    for (int d = 0; d < pair_key_rank; ++d) pair_key[d] = outer_coord[d];
    long min_inner = pair_range.at(pair_key).first;
    for (auto& [inner_coord, val] : entries)
      for (int d = 0; d < inner_rank; ++d) inner_coord[d] -= min_inner;
  }

  // Outer TiledRange: pair-key dims tiled singly, the rest adaptive.
  //
  // REVERTED (see git history / results/README.md "eq64 hang" section for
  // the full story): an earlier version of this pass tried coarsening the
  // pair-key dims to adaptive tile sizes too, moving the pair-key lookup
  // from once-per-tile to once-per-outer-element, on the theory that
  // TA::Tensor<Tensor<double>> supports a ragged inner size per outer
  // position so a tile straddling multiple pairs would still be correct.
  // That theory was WRONG in a way the ethane/propane/butane correctness
  // check (matching checksums against the pre-change baseline) never
  // caught, because on those molecules the occupied dim is small enough
  // that adaptive_tile_sizes() (target ~8 tiles/dim) already rounds down
  // to tile size 1 for the pair-key dims anyway — so the "coarsened"
  // tiling was accidentally identical to size-1 tiling for all three, and
  // the change looked safe. Pentane's occupied dim (21) is the first size
  // where adaptive tiling actually produces tile size >1 (21/8=2) for a
  // pair-key dim, and that's exactly where TA::einsum crashed (SIGSEGV,
  // invalid-permissions fault, on eq10) — confirming a genuinely
  // multi-pair outer tile is NOT safe to feed to TA::einsum here, matching
  // the ORIGINAL design comment's claim (mirroring MPQC's own
  // `pao_to_pno_mp2.ipp` `init_tiles_nested` constraint) that a tile must
  // not straddle two pairs. Back to forcing pair-key dims to tile size 1;
  // the per-element (rather than per-tile) pair lookup below is kept
  // since it's equivalent and slightly clearer when tile size is 1 anyway.
  std::vector<std::size_t> outer_shape(coo.shape.begin(), coo.shape.begin() + outer_rank);
  auto full_adaptive = adaptive_tile_sizes(coo.shape, outer_rank);
  TA::TiledRange outer_trange;
  if (use_coarse) {
    // Pair-key (occupied) dims are coarsened to occ_tile by the adaptive
    // override (extent==occ_ext -> occ_tile); non-pair dims keep adaptive.
    outer_trange = make_trange(outer_shape, full_adaptive);
  } else {
    std::vector<std::size_t> outer_tile_sizes(outer_rank);
    for (int d = 0; d < outer_rank; ++d)
      outer_tile_sizes[d] = (d < pair_key_rank) ? 1 : full_adaptive[d];
    outer_trange = make_trange(outer_shape, outer_tile_sizes);
  }
  const auto& tiles_range = outer_trange.tiles_range();

  // Coarse mode: derive the padded-uniform per-tile inner size (max PNO
  // count over the pairs a tile spans) from this array's own pair_range,
  // keyed by the pair-key tile multi-index — exactly what the sidecar's
  // tile_pad_volume supplies, so the shared inner_range_fn (use_padded)
  // path below is reused verbatim.
  if (use_coarse && coarse_pad) {
    for (const auto& [pk, lohi] : pair_range) {
      std::vector<long> rep(outer_rank, 0);
      for (int d = 0; d < pair_key_rank; ++d) rep[d] = pk[d];
      auto ord = outer_trange.element_to_tile(rep);
      auto midx = tiles_range.idx(ord);
      std::vector<long> pad_key(pair_key_rank);
      for (int d = 0; d < pair_key_rank; ++d)
        pad_key[d] = static_cast<long>(midx[d]);
      long npno = lohi.second - lohi.first + 1;
      auto it = tile_pad_volume.find(pad_key);
      if (it == tile_pad_volume.end() ||
          static_cast<long>(it->second) < npno)
        tile_pad_volume[pad_key] = static_cast<std::size_t>(npno);
    }
  }

  // Pass 1: per-outer-tile norms (sum of squares over every cell's every
  // inner element in that tile) for the outer SparseShape.
  TA::Tensor<float> tile_norms(tiles_range, 0.0f);
  for (auto& [outer_coord, entries] : data_by_outer) {
    auto tidx = outer_trange.element_to_tile(outer_coord);
    float acc = 0.0f;
    for (auto& [inner_coord, val] : entries) {
      float v = static_cast<float>(val);
      acc += v * v;
    }
    tile_norms[tidx] += acc;
  }
  tile_norms.inplace_unary([](float& x) { x = std::sqrt(x); });

  TA::SparseShape<float> sp_shape(world, tile_norms, outer_trange);
  ArrayToT array(world, outer_trange, sp_shape);

  // Pass 2: build + set local non-zero outer tiles via TiledArray's own
  // type-agnostic two-pass constructor (DistArray::init_tiles_nested) —
  // works uniformly whether InnerT is a plain TA::Tensor<double> or an
  // arena-backed TA::ArenaTensor<double> (arena cells are a non-owning
  // view with no (range, values) constructor, so they can only be built
  // via this kind of size-then-fill protocol; init_tiles_nested handles
  // the size-then-fill split for BOTH cell kinds without any branching
  // here — see TiledArray's own tests/tot_construction.cpp for the
  // precedent this mirrors). Iterates every local, non-zero outer tile
  // internally, same coverage as the old manual array.begin()/end() loop.
  using InnerRange = typename InnerT::range_type;
  array.init_tiles_nested(
      // inner_range_fn: size (or null-out) this outer element's inner cell.
      // NOTE: must return the CELL's own native range type (InnerRange) —
      // TA::Tensor<double>::range_type is TA::Range, but
      // ArenaTensor<double>::range_type is btas::zb::RangeNd<>, which has
      // no conversion from TA::Range. Both support construction from a
      // plain extent container (std::vector<size_t>), so build the extents
      // vector once and let InnerRange's own container constructor do the
      // right thing for whichever cell type this instantiation uses.
      [&](const TA::Range::index_type& oidx) -> InnerRange {
        std::vector<long> outer_coord;
        for (auto v : oidx) outer_coord.push_back(static_cast<long>(v));
        std::vector<long> pair_key(pair_key_rank);
        for (int d = 0; d < pair_key_rank; ++d) pair_key[d] = outer_coord[d];
        auto pr_it = pair_range.find(pair_key);
        long n_pno = (pr_it != pair_range.end())
                         ? (pr_it->second.second - pr_it->second.first + 1)
                         : 0;

        // With real tiling, every position in this element's TILE shares
        // one padded inner size (the tile's own max PNO count across
        // whichever pair-keys it spans) — falls back to this position's
        // own n_pno if the tile unexpectedly has no sidecar entry (e.g.
        // an empty frozen-orbital tile that should never reach this
        // branch since init_tiles_nested only visits non-zero tiles).
        std::size_t inner_size;
        if (use_padded) {
          auto ord = outer_trange.element_to_tile(outer_coord);
          auto tile_multi_idx = tiles_range.idx(ord);
          std::vector<long> pad_key(pair_key_rank);
          for (int d = 0; d < pair_key_rank; ++d)
            pad_key[d] = static_cast<long>(tile_multi_idx[d]);
          auto pit = tile_pad_volume.find(pad_key);
          inner_size = (pit != tile_pad_volume.end())
                           ? pit->second
                           : static_cast<std::size_t>(std::max<long>(n_pno, 0));
        } else {
          inner_size = static_cast<std::size_t>(std::max<long>(n_pno, 0));
        }

        if (inner_size == 0) return InnerRange{};  // deliberately-null cell
        std::vector<std::size_t> inner_extents(inner_rank, inner_size);
        return InnerRange(inner_extents);
      },
      // inner_fill_fn: only invoked for non-null cells (inner_size > 0).
      // NOTE: unlike TA::Tensor<double>::operator[] (which accepts a
      // multi-dimensional coordinate directly), ArenaTensor::operator[]
      // only accepts a flat ordinal — convert via its own range's
      // ordinal() first (same convention as TA::Range::ordinal()).
      [&](InnerT& cell, const TA::Range::index_type& oidx) {
        std::vector<long> outer_coord;
        for (auto v : oidx) outer_coord.push_back(static_cast<long>(v));
        auto found = data_by_outer.find(outer_coord);
        if (found != data_by_outer.end()) {
          for (auto& [inner_coord, val] : found->second)
            cell[cell.range().ordinal(inner_coord)] = val;
        }
      });

  world.gop.fence();

  if (world.rank() == 0 && !label.empty()) {
    std::cerr << "  " << label << " (ToT): outer shape (";
    for (int d = 0; d < outer_rank; ++d) {
      if (d) std::cerr << ",";
      std::cerr << outer_shape[d];
    }
    std::cerr << "), " << pair_range.size() << " distinct pair keys, nnz="
              << coo.values.size() << ", sparsity=" << (sp_shape.sparsity() * 100.0)
              << "%\n";
  }
  dump_pmap_distribution(world, array, label);

  return array;
}

#endif  // SPTC_TA_BUILDER_H
