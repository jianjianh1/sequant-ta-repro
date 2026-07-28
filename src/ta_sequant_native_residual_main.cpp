// Phase 4/5 (twinkly-dazzling-shamir.md): drives the native-SeQuant-
// generated closed-shell CSV-CCSD T1/T2 residual functions
// (generated_t1_residual.cpp / generated_t2_residual.cpp,
// ta_sequant_native_residual.h) against real leaf data loaded via the
// existing TATensors/load_ta_tensors() infrastructure -- no new loading
// code, per Phase 4's design.
//
// Phase C (performance-parity plan, 2026-07-20): emits the same
// TAStageResult/CSV schema as ta_benchmark_main.cpp (shared via ta_stage.h)
// so results are directly comparable to the other drivers and to real
// MPQC's own logged timing. Timing wraps the EXISTING fence points already
// bracketing each compute_t{1,2}_residual_native() call -- no new
// synchronization is added, so the measurement doesn't perturb what it's
// measuring. `submit_s` = time to the call's own return (task submission +
// local work only, per TA::einsum's lazy-evaluation semantics);
// `wall_s` = time additionally through the post-call fence (the honest,
// end-to-end metric -- see ta_stage.h's own comment on TAStageResult).
#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>

#include "ta_dumper.h"
#include "ta_sequant_native_residual.h"
#include "ta_stage.h"
#include "ta_tensor_loader.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Pin MKL to a single thread before TA init, matching ta_benchmark_main.cpp's
// convention, so TA's own task scheduler owns all cores rather than
// oversubscribing against a multi-threaded BLAS -- otherwise a thread-count
// mismatch could masquerade as an algorithmic performance gap. Symbol is
// weak -- only called when linked against MKL.
extern "C" __attribute__((weak)) void MKL_Set_Num_Threads(int);

using Clock = std::chrono::steady_clock;
static double elapsed_s(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

// Times one compute_t{1,2}_residual_native() call and appends a
// TAStageResult built from its checksum. `compute` must return the
// ToT-typed residual; `world` must already be fenced before this is called
// (so the pre-call timer isn't charged for unrelated prior work).
template <typename ComputeFn>
static void time_residual_stage(TA::World& world, const std::string& equation,
                                const TATensors& ts, ComputeFn&& compute,
                                std::vector<TAStageResult>& results) {
  auto t0 = Clock::now();
  ArrayToT r = compute(ts);
  auto t_submit = Clock::now();
  world.gop.fence();
  auto t_wall = Clock::now();

  ChecksumResult cs = ta_compute_checksum(world, r);

  TAStageResult res;
  res.equation = equation;
  res.stage = "whole_residual";
  res.wall_s = elapsed_s(t0, t_wall);
  res.submit_s = elapsed_s(t0, t_submit);
  res.sparsity = r.shape().sparsity();
  res.peak_rss_kb = get_peak_rss_kb();
  res.nnz = cs.nnz;
  res.sum = cs.sum;
  res.sumsq = cs.sumsq;
  res.max_abs = cs.max_abs;
  results.push_back(res);

  std::cout << equation << " residual checksum: nnz=" << cs.nnz
            << " sum=" << cs.sum << " sumsq=" << cs.sumsq
            << " max_abs=" << cs.max_abs << " wall_s=" << res.wall_s
            << " submit_s=" << res.submit_s << "\n";
}

int main(int argc, char** argv) {
  if (MKL_Set_Num_Threads) MKL_Set_Num_Threads(1);
  TA::World& world = TA_SCOPED_INITIALIZE(argc, argv);

  // SPTC_SPARSE_THRESHOLD: override TA's global block-screening threshold.
  // Different contraction factorizations screen different blocks; lowering
  // this recovers a factorization-invariant residual (see MPQC's own
  // per-batch threshold scaling, cck.ipp).
  if (const char* v = std::getenv("SPTC_SPARSE_THRESHOLD")) {
    world.gop.fence();
    TA::SparseShape<float>::threshold(static_cast<float>(std::atof(v)));
  }

  // EXPERIMENT (2026-07-22, performance-parity investigation, Phase M):
  // MADNESS's ThreadPool defaults every idle worker thread to an
  // uncapped busy-spin (WaitPolicy::Busy) on ONE shared, spinlock-
  // protected task queue (confirmed via direct source read, see plan
  // twinkly-dazzling-shamir.md/optimized-gathering-barto.md) -- this env
  // var opts into testing the alternative policies without touching any
  // other tool's default behavior.
  if (const char* v = std::getenv("SPTC_MAD_WAIT_POLICY")) {
    const std::string policy = v;
    int sleep_us = 0;
    if (const char* us = std::getenv("SPTC_MAD_WAIT_SLEEP_US")) sleep_us = std::atoi(us);
    if (policy == "yield") {
      madness::threadpool_wait_policy(madness::WaitPolicy::Yield);
    } else if (policy == "sleep") {
      madness::threadpool_wait_policy(madness::WaitPolicy::Sleep, sleep_us);
    } else if (policy == "busy") {
      madness::threadpool_wait_policy(madness::WaitPolicy::Busy);
    }
  }

  if (argc < 2) {
    std::cerr << "Usage: ta_sequant_native_residual_main <data_dir>\n"
              << "  Options (via env vars):\n"
              << "    SPTC_TRIALS=N   number of timed trials (default: 3)\n"
              << "    SPTC_WARMUP=0|1 run one untimed warmup pass (default: 1)\n"
              << "    SPTC_MAD_WAIT_POLICY=busy|yield|sleep  MADNESS ThreadPool wait policy (default: busy)\n"
              << "    SPTC_MAD_WAIT_SLEEP_US=N  sleep duration in us for 'sleep' policy (default: 0)\n"
              << "    MAD_NUM_THREADS=N  MADNESS's OWN env var (not ours), total app threads incl.\n"
              << "      main -- default is hardware_concurrency() (16 on this box's 8-core/2-way-SMT\n"
              << "      topology). Phase R (2026-07-22) found MAD_NUM_THREADS=10 ~13-23% FASTER than\n"
              << "      the 16-thread default -- MADNESS's single shared spinlock-protected task\n"
              << "      queue means extra SMT-count threads add pure contention, not real parallel\n"
              << "      capacity, once real work (post-CSE) is this fine-grained.\n"
              << "    CPU AFFINITY (2026-07-26, performance-parity investigation, supersedes the\n"
              << "      MAD_NUM_THREADS=10 finding above): pin the whole process to this machine's\n"
              << "      PHYSICAL cores (avoid SMT siblings) via `taskset -c <physical-core-list>`,\n"
              << "      e.g. `taskset -c 0-7` on this 8-physical-core box, AND set\n"
              << "      MAD_NUM_THREADS to the physical core count exactly (8 here, not 10) --\n"
              << "      an 8-way sweep found pinned+8 beats every unpinned config and every other\n"
              << "      pinned thread count (~20-24% faster than unpinned+10). Once pinning removes\n"
              << "      OS scheduling/migration noise, extra threads past the physical core count\n"
              << "      just re-add contention -- the opposite of the unpinned case, where extra\n"
              << "      threads compensated for that noise. RECOMMENDED best-known combination:\n"
              << "      `taskset -c 0-7` + `MAD_NUM_THREADS=8` + `SPTC_MAD_WAIT_POLICY=yield`\n"
              << "      (adjust the core list/thread count to your own machine's physical core\n"
              << "      count if different).\n";
    return 1;
  }
  const std::string data_dir = argv[1];
  // "molecule" column: last path component of data_dir (matches
  // ta_benchmark_main.cpp's convention for the shared CSV schema).
  std::string mol = data_dir;
  if (!mol.empty() && mol.back() == '/') mol.pop_back();
  auto slash = mol.find_last_of('/');
  if (slash != std::string::npos) mol = mol.substr(slash + 1);

  int num_trials = 3;
  if (const char* v = std::getenv("SPTC_TRIALS")) num_trials = std::atoi(v);
  bool warmup = true;
  if (const char* v = std::getenv("SPTC_WARMUP")) warmup = std::atoi(v) != 0;

  std::cout << "Loading tensors from " << data_dir << "...\n";
  TATensors ts = load_ta_tensors(world, data_dir);
  world.gop.fence();
  std::cout << std::setprecision(15);

  auto run_once = [&](std::vector<TAStageResult>& results) {
    try {
      time_residual_stage(world, "whole_t1_residual", ts,
                          compute_t1_residual_native, results);
    } catch (const std::exception& ex) {
      std::cerr << "T1 residual FAILED: " << ex.what() << "\n";
      print_fail_row(world, mol, "whole_t1_residual", 0, "EXC", ex.what());
    }
    try {
      time_residual_stage(world, "whole_t2_residual", ts,
                          compute_t2_residual_native, results);
    } catch (const std::exception& ex) {
      std::cerr << "T2 residual FAILED: " << ex.what() << "\n";
      print_fail_row(world, mol, "whole_t2_residual", 0, "EXC", ex.what());
    }
  };

  if (warmup) {
    std::cout << "Warmup pass (untimed, discarded)...\n";
    std::vector<TAStageResult> discard;
    run_once(discard);
  }

  print_csv_header(world);
  for (int trial = 1; trial <= num_trials; ++trial) {
    std::cout << "--- trial " << trial << "/" << num_trials << " ---\n";
    std::vector<TAStageResult> results;
    run_once(results);
    print_csv(world, mol, results, trial);
  }

  return 0;
}
