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
// Correctness invariant for TiledArray sparse arrays.
//
// ta_compute_checksum(world, T) — for a flat TSpArrayD or a tensor-of-tensor
// ArrayToT — walks the local non-zero tiles of `T`, computes
// (nnz, sum, sumsq, max_abs) over their stored elements, then MPI-reduces
// across the world. Returns the global tuple, identical on every rank. This
// is how residual output is validated against the reference checksums.
// ---------------------------------------------------------------------------

struct ChecksumResult {
  int64_t nnz = 0;
  double sum = 0.0;
  double sumsq = 0.0;
  double max_abs = 0.0;
};

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

#endif  // SPTC_TA_DUMPER_H
