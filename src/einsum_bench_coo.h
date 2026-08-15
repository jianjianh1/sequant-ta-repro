#ifndef SPTC_EINSUM_BENCH_COO_H
#define SPTC_EINSUM_BENCH_COO_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tiledarray.h>

#include "ta_tensors.h"

struct BenchCOO {
  std::vector<std::size_t> shape;
  std::vector<std::vector<long>> indices;
  std::vector<double> values;
};

inline std::vector<std::size_t> parse_size_list(const std::string& text,
                                                char delimiter = ',') {
  std::vector<std::size_t> result;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, delimiter)) {
    if (!token.empty()) result.push_back(std::stoull(token));
  }
  return result;
}

inline BenchCOO load_bench_coo(const std::string& path,
                               std::vector<std::size_t> shape) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open COO file: " + path);
  if (shape.empty()) throw std::runtime_error("logical shape is empty: " + path);
  BenchCOO result;
  result.shape = std::move(shape);
  std::string line;
  while (std::getline(input, line)) {
    auto first = line.find_first_not_of(" \t");
    if (first == std::string::npos || line[first] == '#') continue;
    std::istringstream row(line);
    std::vector<long> index(result.shape.size());
    for (std::size_t axis = 0; axis < index.size(); ++axis) {
      if (!(row >> index[axis]))
        throw std::runtime_error("short COO row in " + path);
      if (index[axis] < 0 ||
          static_cast<std::size_t>(index[axis]) >= result.shape[axis])
        throw std::runtime_error("COO coordinate outside logical shape in " + path);
    }
    double value;
    if (!(row >> value)) throw std::runtime_error("missing COO value in " + path);
    std::string extra;
    if (row >> extra) throw std::runtime_error("extra field in COO row in " + path);
    result.indices.push_back(std::move(index));
    result.values.push_back(value);
  }
  return result;
}

inline TA::TiledRange1 bench_tr1(std::size_t extent, std::size_t target_tiles) {
  auto tile = std::max<std::size_t>(1, (extent + target_tiles - 1) / target_tiles);
  std::vector<std::size_t> bounds;
  for (std::size_t value = 0; value < extent; value += tile) bounds.push_back(value);
  bounds.push_back(extent);
  return TA::TiledRange1(bounds.begin(), bounds.end());
}

inline TA::TiledRange bench_trange(const std::vector<std::size_t>& shape,
                                   std::size_t target_tiles,
                                   const std::vector<std::size_t>& singleton_axes = {}) {
  std::vector<TA::TiledRange1> dimensions;
  for (std::size_t axis = 0; axis < shape.size(); ++axis)
    dimensions.push_back(bench_tr1(
        shape[axis],
        std::find(singleton_axes.begin(), singleton_axes.end(), axis) !=
                singleton_axes.end()
            ? shape[axis]
            : target_tiles));
  return TA::TiledRange(dimensions.begin(), dimensions.end());
}

inline TA::TSpArrayD build_bench_flat(TA::World& world, const BenchCOO& coo,
                                      std::size_t target_tiles = 8) {
  auto trange = bench_trange(coo.shape, target_tiles);
  auto const& tiles_range = trange.tiles_range();
  TA::Tensor<float> norms(tiles_range, 0.0f);
  for (std::size_t row = 0; row < coo.values.size(); ++row) {
    auto tile_index = trange.element_to_tile(coo.indices[row]);
    float value = static_cast<float>(coo.values[row]);
    norms[tile_index] += value * value;
  }
  norms.inplace_unary([](float& value) { value = std::sqrt(value); });
  TA::SparseShape<float> shape(world, norms, trange);
  TA::TSpArrayD array(world, trange, shape);

  using Row = std::pair<std::vector<long>, double>;
  std::map<std::size_t, std::vector<Row>> local;
  for (std::size_t row = 0; row < coo.values.size(); ++row) {
    auto tile_index = trange.element_to_tile(coo.indices[row]);
    auto ordinal = tiles_range.ordinal(tile_index);
    if (!array.is_zero(ordinal) && array.is_local(ordinal))
      local[ordinal].emplace_back(coo.indices[row], coo.values[row]);
  }
  for (auto& [ordinal, rows] : local) {
    TA::Tensor<double> tile(trange.make_tile_range(ordinal), 0.0);
    for (auto const& [index, value] : rows) tile[index] = value;
    array.set(ordinal, std::move(tile));
  }
  // At threshold zero SparseShape may conservatively retain a zero-norm tile.
  // Every local nonzero-shape tile still needs a fulfilled future or a later
  // einsum fence will wait forever.
  for (auto it = array.begin(); it != array.end(); ++it) {
    auto ordinal = it.ordinal();
    if (array.is_local(ordinal) && !array.is_zero(ordinal) && !local.count(ordinal))
      array.set(ordinal, TA::Tensor<double>(trange.make_tile_range(ordinal), 0.0));
  }
  world.gop.fence();
  return array;
}

using RaggedExtents = std::map<std::vector<long>, std::vector<std::size_t>>;

inline RaggedExtents load_ragged_extents(const std::string& path,
                                         std::size_t pair_rank,
                                         std::size_t inner_rank) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open ragged extent file: " + path);
  RaggedExtents result;
  std::string line;
  while (std::getline(input, line)) {
    auto first = line.find_first_not_of(" \t");
    if (first == std::string::npos || line[first] == '#') continue;
    std::istringstream row(line);
    std::vector<long> key(pair_rank);
    std::vector<std::size_t> extents(inner_rank);
    for (auto& value : key)
      if (!(row >> value)) throw std::runtime_error("short ragged key row in " + path);
    for (auto& value : extents)
      if (!(row >> value) || value == 0)
        throw std::runtime_error("invalid ragged extent row in " + path);
    result.emplace(std::move(key), std::move(extents));
  }
  return result;
}

inline ArrayToT build_bench_nested(TA::World& world, const BenchCOO& coo,
                                   std::size_t outer_rank,
                                   std::size_t inner_rank,
                                   const std::vector<std::size_t>& pair_key_axes,
                                   const RaggedExtents& ragged,
                                   std::size_t target_tiles = 8) {
  if (outer_rank + inner_rank != coo.shape.size() || pair_key_axes.empty() ||
      std::any_of(pair_key_axes.begin(), pair_key_axes.end(),
                  [outer_rank](auto axis) { return axis >= outer_rank; }))
    throw std::runtime_error("inconsistent nested ranks");
  std::vector<std::size_t> outer_shape(coo.shape.begin(), coo.shape.begin() + outer_rank);
  auto outer_trange = bench_trange(outer_shape, target_tiles, pair_key_axes);
  auto const& tiles_range = outer_trange.tiles_range();

  using InnerRow = std::pair<std::vector<long>, double>;
  std::map<std::vector<long>, std::vector<InnerRow>> by_outer;
  TA::Tensor<float> norms(tiles_range, 0.0f);
  for (std::size_t row = 0; row < coo.values.size(); ++row) {
    std::vector<long> outer(coo.indices[row].begin(), coo.indices[row].begin() + outer_rank);
    std::vector<long> inner(coo.indices[row].begin() + outer_rank, coo.indices[row].end());
    by_outer[outer].emplace_back(std::move(inner), coo.values[row]);
    auto tile_index = outer_trange.element_to_tile(outer);
    float value = static_cast<float>(coo.values[row]);
    norms[tile_index] += value * value;
  }
  norms.inplace_unary([](float& value) { value = std::sqrt(value); });
  TA::SparseShape<float> sparse_shape(world, norms, outer_trange);
  ArrayToT array(world, outer_trange, sparse_shape);
  using Tile = ArrayToT::value_type;
  using Inner = Tile::value_type;
  using InnerRange = Inner::range_type;
  array.init_tiles_nested(
      [&](TA::Range::index_type const& outer_index) -> InnerRange {
        std::vector<long> key(pair_key_axes.size());
        for (std::size_t position = 0; position < pair_key_axes.size(); ++position)
          key[position] = outer_index[pair_key_axes[position]];
        auto found = ragged.find(key);
        if (found == ragged.end()) return InnerRange{};
        return InnerRange(found->second);
      },
      [&](Inner& cell, TA::Range::index_type const& outer_index) {
        std::vector<long> outer(outer_index.begin(), outer_index.end());
        auto found = by_outer.find(outer);
        if (found == by_outer.end()) return;
        for (auto const& [inner, value] : found->second)
          cell[cell.range().ordinal(inner)] = value;
      });
  world.gop.fence();
  return array;
}

#endif  // SPTC_EINSUM_BENCH_COO_H
