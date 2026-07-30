// gap_microbench.cpp — achievable-speedup CEILING for the cold-path T2
// hotspot (generated_t2_residual.cpp:487-488, the CSE37 block).
//
// The cold-dominant op is the ToT×ToT contraction over the OUTER PAO index
// μ̃ that forms CSE37. TA_EINSUM_INSTRUMENT shows it is 83-85% LocalKernel
// (many tiny per-pair GEMMs), not task/retile machinery. This tool answers:
// how much faster is the SAME math expressed as a few big dense GEMMs per
// occupied pair?
//
//   L487: I_ap2_μ̃_Κ[i1,i2,μ̃_out,Κ ; a]  = Σ_{μ̃_in} g0[μ̃_in,μ̃_out,Κ]
//                                                       · C[i1,i2,μ̃_in ; a]
//   L488: CSE37[i2,i1,Κ ; a1,a3]          = Σ_{μ̃_out} I[i1,i2,μ̃_out,Κ ; a1]
//                                                       · C[i1,i2,μ̃_out ; a3]
//
// (i1,i2) is a shared/batched outer index — per occupied pair the two
// contractions are two dense GEMMs (derivation in the block comments below).
//
// Baseline: the two TA::einsum lines copied verbatim, fenced, timed.
// Ceiling : per-pair BLAS gemm on the same data. Both produce the CSE37
// checksum (nnz,sum,sumsq,max_abs); they must agree to ~10 sig figs, which
// proves the hand-GEMM does the same math.
//
// Build under SPTC_BUILD_TOOLS with -DSPTC_OWNING_TOT (single-node,
// thread-safe). Run single rank, e.g.:
//   mpirun -np 1 --bind-to none -x SPTC_COARSE_OCC=9 -x SPTC_OCC_TILE=2 \
//     -x SPTC_COARSE_PAD=0 -x SPTC_TILES_PER_DIM=8 -x MAD_NUM_THREADS=8 \
//     ./gap_microbench <leafdir>

#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>

#include <blas.hh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "ta_dumper.h"
#include "ta_tensor_loader.h"
#include "ta_tensors.h"

// OpenBLAS thread control (weak: resolved when linked against OpenBLAS).
extern "C" __attribute__((weak)) void openblas_set_num_threads(int);
// MKL, if that were the BLAS instead (kept for portability).
extern "C" __attribute__((weak)) void MKL_Set_Num_Threads(int);

using Clock = std::chrono::steady_clock;
static double elapsed_s(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// Running (nnz,sum,sumsq,max_abs) accumulator — the same tuple
// ta_compute_checksum builds, so the two are directly comparable.
struct Accum {
  int64_t nnz = 0;
  double sum = 0.0, sumsq = 0.0, max_abs = 0.0;
  void add(double v) {
    if (v == 0.0) return;  // matches kTaDumpZeroEps = 0.0 in ta_dumper.h
    ++nnz;
    sum += v;
    sumsq += v * v;
    double av = std::abs(v);
    if (av > max_abs) max_abs = av;
  }
};

// ---------------------------------------------------------------------------
// Extract a FLAT sparse array into a dense row-major buffer indexed by its
// global element coordinates. Single-rank: every non-zero tile is local.
// ---------------------------------------------------------------------------
static std::vector<double> dense_from_flat(const TA::TSpArrayD& A,
                                           std::vector<std::size_t>& extents) {
  const auto& er = A.trange().elements_range();
  const std::size_t rank = er.rank();
  extents.assign(rank, 0);
  std::size_t total = 1;
  for (std::size_t d = 0; d < rank; ++d) {
    extents[d] = er.extent(d);
    total *= extents[d];
  }
  std::vector<double> dense(total, 0.0);
  // row-major strides
  std::vector<std::size_t> stride(rank, 1);
  for (int d = static_cast<int>(rank) - 2; d >= 0; --d)
    stride[d] = stride[d + 1] * extents[d + 1];

  for (auto it = A.begin(); it != A.end(); ++it) {
    auto ord = it.ordinal();
    if (A.is_zero(ord) || !A.is_local(ord)) continue;
    const auto& tile = A.find_local(ord).get();
    const auto& r = tile.range();
    const std::size_t vol = r.volume();
    for (std::size_t k = 0; k < vol; ++k) {
      auto idx = r.idx(k);  // global coords
      std::size_t off = 0;
      for (std::size_t d = 0; d < rank; ++d)
        off += static_cast<std::size_t>(idx[d]) * stride[d];
      dense[off] = tile.data()[k];
    }
  }
  return dense;
}

// One occupied pair's dense C block: matrix [Nm × npno] row-major
// (row = μ̃ PAO index, col = pair-local PNO index a).
struct PairC {
  int npno = 0;
  std::vector<double> mat;  // Nm * npno, row-major
};

// ---------------------------------------------------------------------------
// Extract c2_tot (outer (i1,i2,μ̃), inner a) into per-pair dense C matrices.
// Keyed by (i1,i2). Row μ̃ that carries no data stays zero (correct: it
// contributes nothing to the contraction over μ̃).
// ---------------------------------------------------------------------------
static std::map<std::pair<long, long>, PairC> extract_pairC(
    const ArrayToT& C, std::size_t Nm) {
  std::map<std::pair<long, long>, PairC> out;
  for (auto it = C.begin(); it != C.end(); ++it) {
    auto ord = it.ordinal();
    if (C.is_zero(ord) || !C.is_local(ord)) continue;
    const auto& outer = C.find_local(ord).get();
    const auto& orange = outer.range();
    const std::size_t ovol = orange.volume();
    for (std::size_t k = 0; k < ovol; ++k) {
      const auto& inner = outer.data()[k];
      const std::size_t npno = inner.range().volume();
      if (npno == 0) continue;
      auto oidx = orange.idx(k);  // (i1,i2,μ̃) global
      long i1 = static_cast<long>(oidx[0]);
      long i2 = static_cast<long>(oidx[1]);
      long mu = static_cast<long>(oidx[2]);
      auto key = std::make_pair(i1, i2);
      auto& pc = out[key];
      if (pc.npno == 0) {
        pc.npno = static_cast<int>(npno);
        pc.mat.assign(Nm * npno, 0.0);
      }
      // inner is a length-npno vector over the pair-local PNO index a.
      double* row = pc.mat.data() + static_cast<std::size_t>(mu) * npno;
      for (std::size_t a = 0; a < npno; ++a) row[a] = inner.data()[a];
    }
  }
  return out;
}

int main(int argc, char** argv) {
  if (MKL_Set_Num_Threads) MKL_Set_Num_Threads(1);
  TA::World& world = TA_SCOPED_INITIALIZE(argc, argv);
  if (argc < 2) {
    if (world.rank() == 0)
      std::cerr << "Usage: gap_microbench <leaf_dir>\n";
    return 1;
  }
  const std::string data_dir = argv[1];

  // Threads for the hand-GEMM path: match the residual runs' MAD_NUM_THREADS
  // so the BLAS ceiling and the TA baseline get the same core budget.
  int nthreads = static_cast<int>(std::thread::hardware_concurrency());
  if (const char* v = std::getenv("MAD_NUM_THREADS")) {
    int n = std::atoi(v);
    if (n > 0) nthreads = n;
  }

  int ta_trials = 3;
  if (const char* v = std::getenv("SPTC_TRIALS")) ta_trials = std::atoi(v);

  if (world.rank() == 0)
    std::cout << "Loading tensors from " << data_dir << " ...\n";
  TATensors ts = load_ta_tensors(world, data_dir);
  world.gop.fence();

  // Bind the operand names to match generated_t2_residual.cpp L487-488.
  const TA::TSpArrayD& g_μ̃_μ̃_Κ = ts.g0;
  const ArrayToT& C_ap2_μ̃ = ts.c2_tot;
  const ArrayToT& C_μ̃_ap2 = ts.c2_tot;

  // ---- dimensions ----
  const auto& g_er = g_μ̃_μ̃_Κ.trange().elements_range();
  const std::size_t Nm_in = g_er.extent(0);   // μ̃'
  const std::size_t Nm = g_er.extent(1);       // μ̃  (== Nm_in)
  const std::size_t NK = g_er.extent(2);       // Κ
  const std::size_t Nm_c = C_ap2_μ̃.trange().elements_range().extent(2);
  if (world.rank() == 0) {
    std::cout << "dims: g0 (μ̃'=" << Nm_in << ", μ̃=" << Nm << ", Κ=" << NK
              << "), C μ̃=" << Nm_c << "\n";
  }
  if (Nm_in != Nm || Nm_c != Nm) {
    if (world.rank() == 0)
      std::cerr << "FATAL: μ̃ extent mismatch (g0 " << Nm_in << "/" << Nm
                << ", C " << Nm_c << ")\n";
    return 2;
  }

  // =========================================================================
  // Baseline: the two TA::einsum lines, verbatim (variable names bound above).
  // Warmup once, then time `ta_trials` fenced trials; keep the best (min).
  // =========================================================================
  ArrayToT CSE37_i_i_ap2_ap2_Κ;
  ChecksumResult ta_cs;
  double ta_best = 0.0;
  {
    auto run_ta = [&]() -> double {
      ArrayToT I_ap2_μ̃_Κ;
      ArrayToT CSE37;
      world.gop.fence();
      auto t0 = Clock::now();
      I_ap2_μ̃_Κ("i_1,i_2,μ̃_19906,Κ_1;a_1") = TA::einsum(g_μ̃_μ̃_Κ("μ̃_19905,μ̃_19906,Κ_1"), C_ap2_μ̃("i_1,i_2,μ̃_19905;a_1"), "i_1,i_2,μ̃_19906,Κ_1;a_1")("i_1,i_2,μ̃_19906,Κ_1;a_1");
      CSE37("i_2,i_1,Κ_1;a_1,a_3") = TA::einsum(I_ap2_μ̃_Κ("i_1,i_2,μ̃_19906,Κ_1;a_1"), C_μ̃_ap2("i_1,i_2,μ̃_19906;a_3"), "i_2,i_1,Κ_1;a_1,a_3")("i_2,i_1,Κ_1;a_1,a_3");
      world.gop.fence();
      double s = elapsed_s(t0, Clock::now());
      CSE37_i_i_ap2_ap2_Κ = CSE37;  // keep last for checksum
      return s;
    };
    // warmup
    run_ta();
    for (int t = 0; t < ta_trials; ++t) {
      double s = run_ta();
      if (t == 0 || s < ta_best) ta_best = s;
    }
    ta_cs = ta_compute_checksum(world, CSE37_i_i_ap2_ap2_Κ);
  }

  // =========================================================================
  // Ceiling: per-pair dense BLAS GEMM.
  //
  // Data marshaling (extract g0 to dense, C to per-pair matrices) is done
  // ONCE and EXCLUDED from the timer — the ceiling is the pure GEMM compute
  // (an ideal impl amortizes/streams marshaling; what we bound here is the
  // floor on the arithmetic itself).
  //
  // Layout (all row-major):
  //   gd   : g0 dense, [Nm(μ̃_in) × (Nm(μ̃_out)·NK)], gd[μ̃_in·(Nm·NK)+μ̃_out·NK+Κ]
  //          == the "GB" matrix [μ̃_in × (μ̃_out·NK+Κ)] with no reorder.
  //   cp   : per pair, [Nm(μ̃) × np], cp[μ̃·np + a].
  //   Ip   : L487 output, [(Nm·NK) × np], Ip[(μ̃_out·NK+Κ)·np + a]
  //          == reinterpretable as [Nm(μ̃_out) × (NK·np)] for L488.
  //   R    : CSE37, [(NK·np) × np], R[(Κ·np+a1)·np + a3] = CSE37[Κ,a1,a3].
  //
  //   L487:  Ip = gd^T · cp
  //          C(M×N)=op(A)·op(B), M=Nm·NK, N=np, K=Nm; A=gd stored [K×M],
  //          transA=Trans, lda=Nm·NK; B=cp [K×N] NoTrans, ldb=np; ldc=np.
  //   L488:  R  = Ip^T · cp   (Ip viewed [Nm(μ̃_out) × (NK·np)])
  //          M=NK·np, N=np, K=Nm; A=Ip stored [K×M], transA=Trans,
  //          lda=NK·np; B=cp [K×N] NoTrans, ldb=np; ldc=np.
  // =========================================================================
  if (openblas_set_num_threads) openblas_set_num_threads(nthreads);

  std::vector<std::size_t> g_ext;
  std::vector<double> gd = dense_from_flat(g_μ̃_μ̃_Κ, g_ext);
  auto pairs = extract_pairC(C_ap2_μ̃, Nm);

  // scratch buffers sized to the largest pair
  int max_np = 0;
  for (auto& [k, pc] : pairs) max_np = std::max(max_np, pc.npno);
  std::vector<double> Ip(static_cast<std::size_t>(Nm) * NK * max_np);
  std::vector<double> R(static_cast<std::size_t>(NK) * max_np * max_np);

  Accum hand;
  double total_flops = 0.0;
  int npairs = 0;

  world.gop.fence();
  auto tg0 = Clock::now();
  for (auto& [key, pc] : pairs) {
    const int np = pc.npno;
    if (np == 0) continue;
    ++npairs;
    const double* cp = pc.mat.data();

    // L487: Ip[(Nm·NK) × np] = gd^T · cp
    blas::gemm(blas::Layout::RowMajor, blas::Op::Trans, blas::Op::NoTrans,
               /*m=*/static_cast<int64_t>(Nm * NK), /*n=*/np,
               /*k=*/static_cast<int64_t>(Nm),
               1.0, gd.data(), /*lda=*/static_cast<int64_t>(Nm * NK),
               cp, /*ldb=*/np,
               0.0, Ip.data(), /*ldc=*/np);

    // L488: R[(NK·np) × np] = Ip^T · cp   (Ip as [Nm × (NK·np)])
    blas::gemm(blas::Layout::RowMajor, blas::Op::Trans, blas::Op::NoTrans,
               /*m=*/static_cast<int64_t>(NK * np), /*n=*/np,
               /*k=*/static_cast<int64_t>(Nm),
               1.0, Ip.data(), /*lda=*/static_cast<int64_t>(NK * np),
               cp, /*ldb=*/np,
               0.0, R.data(), /*ldc=*/np);

    // accumulate CSE37 checksum
    const std::size_t rvol = static_cast<std::size_t>(NK) * np * np;
    for (std::size_t i = 0; i < rvol; ++i) hand.add(R[i]);

    total_flops += 2.0 * (double)(Nm * NK) * np * Nm;   // L487
    total_flops += 2.0 * (double)(NK * np) * np * Nm;   // L488
  }
  double hand_best = elapsed_s(tg0, Clock::now());

  // =========================================================================
  // Report
  // =========================================================================
  if (world.rank() == 0) {
    std::cout << std::setprecision(11);
    std::cout << "\n=== gap_microbench: CSE37 (T2 L487-488) ===\n";
    std::cout << "molecule leaf : " << data_dir << "\n";
    std::cout << "threads (BLAS): " << nthreads << "\n";
    std::cout << "occ pairs      : " << npairs << ", Nm(μ̃)=" << Nm
              << ", NK(Κ)=" << NK << ", max np=" << max_np << "\n\n";

    std::cout << "TA_einsum_s   : " << ta_best << "  (best of " << ta_trials
              << ")\n";
    std::cout << "handGEMM_s    : " << hand_best << "\n";
    std::cout << "ceiling_ratio : " << (ta_best / hand_best) << "  (TA / handGEMM)\n\n";

    double gf_ta = total_flops / ta_best / 1e9;
    double gf_hand = total_flops / hand_best / 1e9;
    std::cout << "total_flops   : " << total_flops << "\n";
    std::cout << "GFLOP/s TA    : " << gf_ta << "\n";
    std::cout << "GFLOP/s hand  : " << gf_hand << "  (node peak ~256 GFLOP/s)\n\n";

    std::cout << std::setprecision(15);
    std::cout << "checksum TA   : nnz=" << ta_cs.nnz << " sum=" << ta_cs.sum
              << " sumsq=" << ta_cs.sumsq << " max_abs=" << ta_cs.max_abs << "\n";
    std::cout << "checksum hand : nnz=" << hand.nnz << " sum=" << hand.sum
              << " sumsq=" << hand.sumsq << " max_abs=" << hand.max_abs << "\n";

    auto rel = [](double a, double b) {
      double d = std::abs(a - b);
      double m = std::max(std::abs(a), std::abs(b));
      return m > 0 ? d / m : d;
    };
    std::cout << std::setprecision(3) << std::scientific;
    std::cout << "rel-diff      : sum=" << rel(ta_cs.sum, hand.sum)
              << " sumsq=" << rel(ta_cs.sumsq, hand.sumsq)
              << " max_abs=" << rel(ta_cs.max_abs, hand.max_abs) << "\n";
    bool ok = rel(ta_cs.sum, hand.sum) < 1e-9 &&
              rel(ta_cs.sumsq, hand.sumsq) < 1e-9 &&
              rel(ta_cs.max_abs, hand.max_abs) < 1e-9;
    std::cout << "MATCH         : " << (ok ? "YES (<1e-9)" : "NO") << "\n";
  }

  return 0;
}
