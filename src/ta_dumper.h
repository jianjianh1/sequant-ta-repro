#ifndef SPTC_TA_DUMPER_H
#define SPTC_TA_DUMPER_H

#include <mpi.h>
#include <tiledarray.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ta_tensors.h"  // ArrayToT

// ---------------------------------------------------------------------------
// Correctness invariants + opt-in tensor dumping for TiledArray sparse arrays.
//
// Two responsibilities:
//
// 1. ta_compute_checksum(world, T)
//    Walks the local non-zero tiles of `T`, computes (nnz, sum, sumsq,
//    max_abs) over their stored elements, then MPI-reduces across the world.
//    Returns the global tuple — identical on every rank.
//
// 2. ta_dump_if_enabled(world, T, eq, stage)
//    If the env var SPTC_DUMP_DIR is set AND the current dump context's
//    trial == 1 (the first timed trial, set by ta_benchmark_main.cpp before
//    each runner call), gathers every non-zero element from every rank to
//    rank 0 and writes a text-COO file to
//      ${SPTC_DUMP_DIR}/${molecule}_${eq}_${stage}.coo
//    matching the input format consumed by coo_loader.h:load_coo so the
//    file round-trips through the loader unchanged.
//
// The dump context (molecule, trial, framework) is set by main once per
// (molecule, trial) instead of plumbing extra args through every runner —
// the 44 generated DLPNO runners would otherwise need a signature change.
// ---------------------------------------------------------------------------

struct ChecksumResult {
  int64_t nnz = 0;
  double sum = 0.0;
  double sumsq = 0.0;
  double max_abs = 0.0;
};

struct TaDumpContext {
  std::string molecule;
  int trial = 0;            // 1-based; the dumper writes only when trial == 1
  std::string framework = "ta";
};

inline TaDumpContext& ta_dump_context() {
  static TaDumpContext ctx;
  return ctx;
}

inline void ta_set_dump_context(TaDumpContext ctx) {
  ta_dump_context() = std::move(ctx);
}

// ---------------------------------------------------------------------------
// Internal: per-rank scalar accumulator over the LOCAL non-zero tiles.
// We treat any element whose |val| < kZeroEps as "not stored" so the count
// matches what `coo_loader.h` would produce on a round-trip (the loader
// drops nothing, but post-einsum tiles can hold exact-zero leftover from
// the sparse-shape mask — counting those as nnz would make CTF vs TA
// diverge artificially).
// ---------------------------------------------------------------------------
constexpr double kTaDumpZeroEps = 0.0;

inline void ta_accumulate_local(const TA::TSpArrayD& T,
                                int64_t& nnz, double& sum, double& sumsq,
                                double& max_abs) {
  for (auto it = T.begin(); it != T.end(); ++it) {
    auto ord = it.ordinal();
    if (T.is_zero(ord) || !T.is_local(ord)) continue;
    auto fut = T.find_local(ord);
    const auto& tile = fut.get();   // block until tile is ready
    const auto n_elem = tile.range().volume();
    for (std::size_t i = 0; i < n_elem; ++i) {
      double v = tile.data()[i];
      if (std::abs(v) <= kTaDumpZeroEps) continue;
      ++nnz;
      sum += v;
      sumsq += v * v;
      double av = std::abs(v);
      if (av > max_abs) max_abs = av;
    }
  }
}

// Same as ta_accumulate_local, but for a tensor-of-tensor array: each outer
// tile's stored elements are themselves inner TA::Tensor<double>s, so this
// walks both levels.
inline void ta_accumulate_local_tot(const ArrayToT& T,
                                    int64_t& nnz, double& sum, double& sumsq,
                                    double& max_abs) {
  for (auto it = T.begin(); it != T.end(); ++it) {
    auto ord = it.ordinal();
    if (T.is_zero(ord) || !T.is_local(ord)) continue;
    auto fut = T.find_local(ord);
    const auto& outer_tile = fut.get();  // block until tile is ready
    const auto n_outer = outer_tile.range().volume();
    for (std::size_t i = 0; i < n_outer; ++i) {
      const auto& inner = outer_tile.data()[i];
      const auto n_inner = inner.range().volume();
      for (std::size_t j = 0; j < n_inner; ++j) {
        double v = inner.data()[j];
        if (std::abs(v) <= kTaDumpZeroEps) continue;
        ++nnz;
        sum += v;
        sumsq += v * v;
        double av = std::abs(v);
        if (av > max_abs) max_abs = av;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// ta_compute_checksum: distributed-safe, identical on every rank on return.
// ---------------------------------------------------------------------------
inline ChecksumResult ta_compute_checksum(TA::World& world,
                                          const TA::TSpArrayD& T) {
  int64_t local_nnz = 0;
  double local_sum = 0.0, local_sumsq = 0.0, local_max = 0.0;
  ta_accumulate_local(T, local_nnz, local_sum, local_sumsq, local_max);

  MPI_Comm comm = world.mpi.comm().Get_mpi_comm();
  ChecksumResult r;
  MPI_Allreduce(&local_nnz, &r.nnz, 1, MPI_INT64_T, MPI_SUM, comm);
  MPI_Allreduce(&local_sum, &r.sum, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&local_sumsq, &r.sumsq, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&local_max, &r.max_abs, 1, MPI_DOUBLE, MPI_MAX, comm);
  return r;
}

inline ChecksumResult ta_compute_checksum(TA::World& world,
                                          const ArrayToT& T) {
  int64_t local_nnz = 0;
  double local_sum = 0.0, local_sumsq = 0.0, local_max = 0.0;
  ta_accumulate_local_tot(T, local_nnz, local_sum, local_sumsq, local_max);

  MPI_Comm comm = world.mpi.comm().Get_mpi_comm();
  ChecksumResult r;
  MPI_Allreduce(&local_nnz, &r.nnz, 1, MPI_INT64_T, MPI_SUM, comm);
  MPI_Allreduce(&local_sum, &r.sum, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&local_sumsq, &r.sumsq, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&local_max, &r.max_abs, 1, MPI_DOUBLE, MPI_MAX, comm);
  return r;
}

// ---------------------------------------------------------------------------
// Internal: collect this rank's non-zero (multi-index, value) pairs into
// flat buffers ready for MPI_Gatherv. Rank R contributes
//   indices_local: int64_t[R_dims * nnz_local]   (row-major, dim-fastest)
//   values_local:  double[nnz_local]
// ---------------------------------------------------------------------------
inline void ta_collect_local_coo(const TA::TSpArrayD& T, int rank,
                                 std::vector<int64_t>& indices_local,
                                 std::vector<double>& values_local) {
  for (auto it = T.begin(); it != T.end(); ++it) {
    auto ord = it.ordinal();
    if (T.is_zero(ord) || !T.is_local(ord)) continue;
    auto fut = T.find_local(ord);
    const auto& tile = fut.get();
    const auto& tile_range = tile.range();
    // tile_range iterates multi-indices in global element coordinates.
    for (auto rit = tile_range.begin(); rit != tile_range.end(); ++rit) {
      const auto& multi_idx = *rit;
      double v = tile[multi_idx];
      if (std::abs(v) <= kTaDumpZeroEps) continue;
      for (int d = 0; d < rank; ++d)
        indices_local.push_back(static_cast<int64_t>(multi_idx[d]));
      values_local.push_back(v);
    }
  }
}

// ---------------------------------------------------------------------------
// ta_dump_array: rank-0-side text-COO writer. Format matches coo_loader.h:
//   First line:  '# shape=d0,d1,...,dN nnz=K rank=R framework=ta '
//                'molecule=... equation=... stage=... trial=...'
//   Then K lines: 'idx0 idx1 ... idxN value\n'
// Only rank 0 touches the filesystem.
// ---------------------------------------------------------------------------
inline void ta_dump_array(TA::World& world, const TA::TSpArrayD& T,
                          const std::string& path,
                          const std::string& header_comment) {
  const int my_rank = world.rank();
  const int nproc = world.size();
  const int tensor_rank = T.trange().rank();
  MPI_Comm comm = world.mpi.comm().Get_mpi_comm();

  std::vector<int64_t> indices_local;
  std::vector<double> values_local;
  ta_collect_local_coo(T, tensor_rank, indices_local, values_local);
  const int64_t nnz_local = static_cast<int64_t>(values_local.size());

  // Gather per-rank counts.
  std::vector<int64_t> nnz_per_rank(nproc, 0);
  MPI_Gather(&nnz_local, 1, MPI_INT64_T,
             nnz_per_rank.data(), 1, MPI_INT64_T, 0, comm);

  // Compute displacements (rank 0 only); also derive total nnz.
  std::vector<int> recv_counts_idx, recv_displs_idx;
  std::vector<int> recv_counts_val, recv_displs_val;
  int64_t total_nnz = 0;
  if (my_rank == 0) {
    recv_counts_idx.resize(nproc);
    recv_displs_idx.resize(nproc);
    recv_counts_val.resize(nproc);
    recv_displs_val.resize(nproc);
    int64_t acc_idx = 0, acc_val = 0;
    for (int r = 0; r < nproc; ++r) {
      total_nnz += nnz_per_rank[r];
      recv_counts_val[r] = static_cast<int>(nnz_per_rank[r]);
      recv_displs_val[r] = static_cast<int>(acc_val);
      acc_val += nnz_per_rank[r];
      recv_counts_idx[r] = static_cast<int>(nnz_per_rank[r] * tensor_rank);
      recv_displs_idx[r] = static_cast<int>(acc_idx);
      acc_idx += nnz_per_rank[r] * tensor_rank;
    }
  }

  // Gather index + value payloads.
  std::vector<int64_t> indices_all;
  std::vector<double> values_all;
  if (my_rank == 0) {
    indices_all.resize(total_nnz * tensor_rank);
    values_all.resize(total_nnz);
  }
  MPI_Gatherv(indices_local.data(),
              static_cast<int>(indices_local.size()), MPI_INT64_T,
              my_rank == 0 ? indices_all.data() : nullptr,
              my_rank == 0 ? recv_counts_idx.data() : nullptr,
              my_rank == 0 ? recv_displs_idx.data() : nullptr,
              MPI_INT64_T, 0, comm);
  MPI_Gatherv(values_local.data(),
              static_cast<int>(values_local.size()), MPI_DOUBLE,
              my_rank == 0 ? values_all.data() : nullptr,
              my_rank == 0 ? recv_counts_val.data() : nullptr,
              my_rank == 0 ? recv_displs_val.data() : nullptr,
              MPI_DOUBLE, 0, comm);

  if (my_rank != 0) return;

  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path());
  std::ofstream out(path);
  if (!out) {
    std::cerr << "WARNING: ta_dump_array failed to open " << path << "\n";
    return;
  }
  // Shape comment line: scan the global tile range and emit per-dim extent.
  out << "# shape=";
  const auto& elements_range = T.trange().elements_range();
  for (int d = 0; d < tensor_rank; ++d) {
    if (d) out << ',';
    out << elements_range.extent(d);
  }
  out << " nnz=" << total_nnz << " rank=" << tensor_rank
      << ' ' << header_comment << '\n';
  out << std::setprecision(17);
  for (int64_t i = 0; i < total_nnz; ++i) {
    for (int d = 0; d < tensor_rank; ++d)
      out << indices_all[i * tensor_rank + d] << ' ';
    out << values_all[i] << '\n';
  }
}

// ---------------------------------------------------------------------------
// ta_dump_if_enabled: convenience wrapper used by stage runners. Honors the
// SPTC_DUMP_DIR env var and the trial-1 gate (avoiding 3x duplication when
// SPTC_TRIALS=3).
// ---------------------------------------------------------------------------
inline void ta_dump_if_enabled(TA::World& world, const TA::TSpArrayD& T,
                               const std::string& eq, const std::string& stage) {
  const char* dump_dir = std::getenv("SPTC_DUMP_DIR");
  if (!dump_dir || !*dump_dir) return;
  const auto& ctx = ta_dump_context();
  if (ctx.trial != 1) return;   // only the first timed trial writes
  std::ostringstream path;
  path << dump_dir << '/' << ctx.molecule << '_' << eq << '_' << stage << ".coo";
  std::ostringstream hdr;
  hdr << "framework=" << ctx.framework << " molecule=" << ctx.molecule
      << " equation=" << eq << " stage=" << stage
      << " nranks=" << world.size() << " trial=" << ctx.trial;
  ta_dump_array(world, T, path.str(), hdr.str());
}

#endif  // SPTC_TA_DUMPER_H
