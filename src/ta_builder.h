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
    std::size_t target_tiles_per_dim = 8) {
  // EXPERIMENT (2026-07-20, performance-parity investigation): allow
  // overriding the target tile count per dimension without touching the
  // shared default (which every ta-bench tool relies on) -- used to test
  // whether fewer/larger tiles reduce TA's per-task scheduling overhead
  // for the native SeQuant generator's ~500-statement whole-residual
  // computation. Not wired into any other tool's behavior unless the env
  // var is explicitly set.
  if (const char* v = std::getenv("SPTC_TILES_PER_DIM")) {
    std::size_t n = static_cast<std::size_t>(std::atoi(v));
    if (n > 0) target_tiles_per_dim = n;
  }
  std::vector<std::size_t> sizes(rank);
  for (int d = 0; d < rank; ++d) {
    std::size_t extent = shape[d];
    sizes[d] = std::max<std::size_t>(1, extent / target_tiles_per_dim);
  }
  return sizes;
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

  return array;
}

/// Real-tiling proof-of-concept (see plan: "Use MPQC's real ToT tile
/// structure to fix ta_benchmark's eq64 hang"). MPQC's own real production
/// `.tile.tns` dumps (structural-only; see
/// `/users/jianjian/mpqc4/bin/tns-to-sptc-coo.py`'s `read_tile_boundaries()`,
/// which generates the sidecars this loads) show MPQC does NOT tile
/// pair-key dims singly — real tiles group 3-4 pair-keys, with DIFFERENT
/// PNO counts sharing one tile, and MPQC doesn't crash. The likely reason
/// this file's own earlier coarsening attempt (see the comment on
/// `build_tot_array()` below) crashed: it computed a distinct inner `Range`
/// per outer POSITION within one tile. This loader instead supplies
/// externally-derived real tile boundaries plus a single PADDED inner size
/// per TILE (every position in a tile shares one Range, zero-padded past
/// its own true count) — the mechanism difference the plan is testing.
struct RealTilingSpec {
  // One boundary list per pair-key dimension, TA::TiledRange1-ready
  // (first entry 0, last entry = that dimension's full extent).
  std::vector<std::vector<std::size_t>> dim_boundaries;
  // {tile multi-index (first pair_key_rank dims only) -> padded inner size}
  std::map<std::vector<long>, std::size_t> tile_pad_volume;
};

/// Parse the plain-text sidecar `tns-to-sptc-coo.py`'s `write_tiling_spec()`
/// emits:
///   # pair_key_rank=<N>
///   # dim <d> boundaries: <b0> <b1> ... <bK>     (one line per dim)
///   <tile_idx_0> ... <tile_idx_{N-1}> <pad_volume>   (one line per tile)
inline RealTilingSpec load_real_tiling_spec(const std::string& path) {
  RealTilingSpec spec;
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open real tiling sidecar: " + path);

  int pair_key_rank = -1;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') {
      if (line.rfind("# pair_key_rank=", 0) == 0) {
        pair_key_rank = std::stoi(line.substr(16));
        spec.dim_boundaries.assign(pair_key_rank, {});
      } else if (line.rfind("# dim ", 0) == 0) {
        int d = std::stoi(line.substr(6));
        auto colon = line.find(':');
        std::istringstream iss(line.substr(colon + 1));
        std::size_t v;
        while (iss >> v) spec.dim_boundaries[d].push_back(v);
      }
      continue;
    }
    if (pair_key_rank < 0)
      throw std::runtime_error(path + ": data row before '# pair_key_rank=' header");
    std::istringstream iss(line);
    std::vector<long> tile_idx(pair_key_rank);
    for (int d = 0; d < pair_key_rank; ++d) iss >> tile_idx[d];
    std::size_t pad_volume;
    iss >> pad_volume;
    spec.tile_pad_volume[tile_idx] = pad_volume;
  }
  return spec;
}

/// Build a tensor-of-tensor (ToT) array from COO data, reconstructing MPQC's
/// real per-(occupied-index-tuple)-restricted PNO/OSV structure — the
/// representation that actually bounds memory for equations combining
/// several simultaneous virtual/PNO indices (see eq64: a flat translation
/// treats each virtual axis as a global ~600-wide dimension, when in the
/// real per-pair-restricted structure it's only ever tens of values wide).
///
/// `outer_rank` leading dims are block-sparse/outer (e.g. i,m for C1);
/// the remaining `inner_rank` trailing dims are the per-pair-restricted
/// virtual/PNO axes (e.g. a for C1). `pair_key_rank` (<= outer_rank) is how
/// many of the LEADING outer dims are the occupied-index tuple that
/// determines a pair's PNO count (1 for C1/T1's single occupied index, 2 for
/// C2/T2's occupied pair).
///
/// By default (`real_tiling_sidecar` empty), pair-key dims are tiled singly
/// (one occupied index per tile) so every outer tile maps to exactly one
/// pair key and therefore one well-defined inner Range. This constraint is
/// load bearing for the DEFAULT construction, not a convenience shortcut: a
/// prior pass here tried coarsening pair-key tiling under the theory that
/// `TA::Tensor<Tensor<double>>` supports a ragged inner size per outer
/// position, so a straddling tile would still be correct. That looked safe
/// on ethane/propane/butane (checksums matched the pre-change baseline)
/// purely because their occupied dim is small enough that adaptive tiling
/// already rounds down to size 1 for those dims regardless — the "fix" was
/// a no-op there. Pentane's occupied dim (21) is the first size where
/// adaptive tiling produces tile size >1 (21/8=2) for a pair-key dim, and
/// that's exactly where `TA::einsum` crashed (SIGSEGV, invalid-permissions
/// fault, eq10) — confirming a per-POSITION-varying inner Range within one
/// tile is unsafe. (NOT, it turns out, proof that a multi-pair tile itself
/// is unsafe — see `RealTilingSpec` above: MPQC's own real tiles group
/// multiple pairs successfully, using one padded, uniform inner Range per
/// TILE instead. `real_tiling_sidecar`, when non-empty, opts into exactly
/// that mechanism instead of the default size-1 path — see
/// `results/README.md`'s "eq64 hang" section for the full investigation.)
///
/// Verified directly against ethane's real per-pair PNO counts: for a given
/// pair key, the virtual index's raw ("global") values form a *contiguous*
/// range, so `a_local = a_global - min_a_for_pair` needs no sidecar file —
/// derived here from one scan of the COO data alone.
template <typename ArrayToT>
inline ArrayToT build_tot_array(TA::World& world, const COOTensor& coo,
                                int outer_rank, int inner_rank,
                                int pair_key_rank,
                                const std::string& label = "",
                                const std::string& real_tiling_sidecar = "") {
  using InnerT = typename ArrayToT::value_type::value_type;  // TA::Tensor<double>

  const bool use_real_tiling = !real_tiling_sidecar.empty();
  RealTilingSpec real_spec;
  if (use_real_tiling) real_spec = load_real_tiling_spec(real_tiling_sidecar);

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
  if (use_real_tiling) {
    // Pair-key dims use MPQC's real tile boundaries (RealTilingSpec, may
    // group multiple pairs per tile); other dims keep the existing
    // adaptive-size approach.
    std::vector<TA::TiledRange1> tr1s(outer_rank);
    for (int d = 0; d < outer_rank; ++d) {
      if (d < pair_key_rank) {
        const auto& b = real_spec.dim_boundaries[d];
        tr1s[d] = TA::TiledRange1(b.begin(), b.end());
      } else {
        tr1s[d] = make_tr1(outer_shape[d], full_adaptive[d]);
      }
    }
    outer_trange = TA::TiledRange(tr1s.begin(), tr1s.end());
  } else {
    std::vector<std::size_t> outer_tile_sizes(outer_rank);
    for (int d = 0; d < outer_rank; ++d)
      outer_tile_sizes[d] = (d < pair_key_rank) ? 1 : full_adaptive[d];
    outer_trange = make_trange(outer_shape, outer_tile_sizes);
  }
  const auto& tiles_range = outer_trange.tiles_range();

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
        if (use_real_tiling) {
          auto ord = outer_trange.element_to_tile(outer_coord);
          auto tile_multi_idx = tiles_range.idx(ord);
          std::vector<long> pad_key(pair_key_rank);
          for (int d = 0; d < pair_key_rank; ++d)
            pad_key[d] = static_cast<long>(tile_multi_idx[d]);
          auto pit = real_spec.tile_pad_volume.find(pad_key);
          inner_size = (pit != real_spec.tile_pad_volume.end())
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

  return array;
}

#endif  // SPTC_TA_BUILDER_H
