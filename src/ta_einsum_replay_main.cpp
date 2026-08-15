#include <TiledArray/expressions/einsum.h>
#include <tiledarray.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "einsum_bench_coo.h"
#include "ta_dumper.h"
#include "ta_stage.h"

extern "C" __attribute__((weak)) void MKL_Set_Num_Threads(int);

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::map<std::string, std::string> values;
  bool has(std::string const& key) const { return values.count(key) != 0; }
  std::string get(std::string const& key, std::string fallback = {}) const {
    auto found = values.find(key);
    return found == values.end() ? fallback : found->second;
  }
  std::size_t number(std::string const& key, std::size_t fallback = 0) const {
    return has(key) ? std::stoull(get(key)) : fallback;
  }
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (!key.starts_with("--") || i + 1 == argc)
      throw std::runtime_error("expected --key value arguments");
    options.values.emplace(key.substr(2), argv[++i]);
  }
  for (auto const& required : {"left", "right", "left-shape", "right-shape",
                               "left-storage", "right-storage", "result-storage",
                               "left-annot", "right-annot", "result-annot",
                               "contraction-id", "family", "suite-id", "molecule",
                               "build-id", "trial", "logical-flops"})
    if (!options.has(required)) throw std::runtime_error("missing --" + std::string(required));
  return options;
}

std::vector<std::string> annotation_labels(std::string text) {
  std::vector<std::string> labels;
  for (char& character : text)
    if (character == ';') character = ',';
  std::stringstream stream(text);
  std::string label;
  while (std::getline(stream, label, ',')) {
    label.erase(std::remove_if(label.begin(), label.end(), [](unsigned char value) {
                  return std::isspace(value);
                }),
                label.end());
    if (!label.empty()) labels.push_back(label);
  }
  return labels;
}

void validate_annotations(BenchCOO const& left, BenchCOO const& right,
                          std::string const& left_annot,
                          std::string const& right_annot) {
  auto left_labels = annotation_labels(left_annot);
  auto right_labels = annotation_labels(right_annot);
  if (left_labels.size() != left.shape.size() ||
      right_labels.size() != right.shape.size())
    throw std::runtime_error("annotation rank does not match operand shape");
  std::map<std::string, std::size_t> extents;
  for (auto const& [labels, shape] :
       {std::pair{left_labels, left.shape}, std::pair{right_labels, right.shape}})
    for (std::size_t axis = 0; axis < labels.size(); ++axis) {
      auto [where, inserted] = extents.emplace(labels[axis], shape[axis]);
      if (!inserted && where->second != shape[axis])
        throw std::runtime_error("extent disagreement for annotation " + labels[axis]);
    }
}

long peak_rss_kb() {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      long value = 0;
      std::sscanf(line.c_str(), "VmHWM: %ld kB", &value);
      return value;
    }
  }
  return 0;
}

ArrayToT load_nested(TA::World& world, Options const& options,
                     std::string const& side, BenchCOO const& coo,
                     std::size_t target_tiles) {
  auto outer = options.number(side + "-outer-rank");
  auto inner = options.number(side + "-inner-rank");
  auto pair = options.number(side + "-pair-key-rank");
  auto pair_axes = options.has(side + "-pair-key-axes")
                       ? parse_size_list(options.get(side + "-pair-key-axes"))
                       : [&] {
                           std::vector<std::size_t> axes(pair);
                           for (std::size_t axis = 0; axis < pair; ++axis) axes[axis] = axis;
                           return axes;
                         }();
  if (pair_axes.size() != pair)
    throw std::runtime_error(side + " pair-key rank/axes disagreement");
  auto ragged = load_ragged_extents(options.get(side + "-ragged"), pair, inner);
  return build_bench_nested(world, coo, outer, inner, pair_axes, ragged, target_tiles);
}

template <typename Compute>
auto timed(TA::World& world, Compute&& compute) {
  world.gop.fence();
  auto start = Clock::now();
  auto result = compute();
  world.gop.fence();
  auto stop = Clock::now();
  return std::pair{std::move(result), std::chrono::duration<double>(stop - start).count()};
}

template <typename Array>
void emit(TA::World& world, Options const& options, Array const& result,
          double wall_s, std::size_t input_nnz) {
  auto checksum = ta_compute_checksum(world, result);
  long rss = peak_rss_kb();
  world.gop.max(rss);
  if (world.rank() != 0) return;
  std::cout << "mpqc-einsum-results/v1"
            << ",tiledarray"
            << "," << options.get("build-id")
            << "," << options.get("suite-id")
            << "," << options.get("molecule")
            << "," << options.get("contraction-id")
            << "," << options.get("family")
            << "," << world.size()
            << ",1"
            << "," << options.get("threads", "8")
            << "," << options.get("trial")
            << ",isolated_no_reuse"
            << "," << std::setprecision(17) << wall_s
            << "," << options.get("logical-flops")
            << "," << input_nnz
            << "," << checksum.nnz
            << "," << rss
            << "," << checksum.nnz
            << "," << checksum.sum
            << "," << checksum.sumsq
            << "," << checksum.max_abs
            << ",OK,\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (MKL_Set_Num_Threads) MKL_Set_Num_Threads(1);
    Options options = parse_options(argc, argv);
    TA::World& world = TA_SCOPED_INITIALIZE(argc, argv);
    auto progress = [&](char const* message) {
      if (world.rank() == 0 && std::getenv("SPTC_REPLAY_VERBOSE"))
        std::cerr << "[replay] " << message << "\n" << std::flush;
    };
    TA::SparseShape<float>::threshold(0.0f);
    progress("loading COO inputs");
    auto left_coo = load_bench_coo(options.get("left"), parse_size_list(options.get("left-shape")));
    auto right_coo = load_bench_coo(options.get("right"), parse_size_list(options.get("right-shape")));
    std::size_t input_nnz = left_coo.values.size() + right_coo.values.size();
    auto target_tiles = options.number("tiles-per-dim", 8);
    auto const& la = options.get("left-annot");
    auto const& ra = options.get("right-annot");
    auto const& oa = options.get("result-annot");
    validate_annotations(left_coo, right_coo, la, ra);
    auto family = options.get("family");

    if (family == "flat_flat_flat") {
      progress("building flat operands");
      auto left = build_bench_flat(world, left_coo, target_tiles);
      auto right = build_bench_flat(world, right_coo, target_tiles);
      progress("executing einsum");
      auto [result, seconds] = timed(world, [&] { return TA::einsum(left(la), right(ra), oa); });
      progress("checksumming result");
      emit(world, options, result, seconds, input_nnz);
    } else if (family == "flat_nested_nested") {
      bool left_nested = options.get("left-storage") == "nested";
      if (left_nested) {
        auto left = load_nested(world, options, "left", left_coo, target_tiles);
        auto right = build_bench_flat(world, right_coo, target_tiles);
        auto [result, seconds] = timed(world, [&] { return TA::einsum(left(la), right(ra), oa); });
        emit(world, options, result, seconds, input_nnz);
      } else {
        auto left = build_bench_flat(world, left_coo, target_tiles);
        auto right = load_nested(world, options, "right", right_coo, target_tiles);
        auto [result, seconds] = timed(world, [&] { return TA::einsum(left(la), right(ra), oa); });
        emit(world, options, result, seconds, input_nnz);
      }
    } else if (family == "nested_nested_nested") {
      auto left = load_nested(world, options, "left", left_coo, target_tiles);
      auto right = load_nested(world, options, "right", right_coo, target_tiles);
      auto [result, seconds] = timed(world, [&] { return TA::einsum(left(la), right(ra), oa); });
      emit(world, options, result, seconds, input_nnz);
    } else if (family == "nested_nested_flat") {
      auto left = load_nested(world, options, "left", left_coo, target_tiles);
      auto right = load_nested(world, options, "right", right_coo, target_tiles);
      auto [result, seconds] = timed(
          world, [&] { return TA::einsum<TA::DeNest::True>(left(la), right(ra), oa); });
      emit(world, options, result, seconds, input_nnz);
    } else {
      throw std::runtime_error("unsupported contraction family: " + family);
    }
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "ta_einsum_replay: " << error.what() << "\n";
    return 2;
  }
}
