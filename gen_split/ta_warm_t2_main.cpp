// Warm-loop T2 driver: persistent cache of the t-independent DF/CSV block
// (MPQC's volatile/persistent split). The t-independent intermediates are
// computed once and kept as locals in run_warm_t2_bench (one scope, no
// cross-boundary struct); the t-dependent residual is timed across trials —
// the fair analog of MPQC's warm iter2 measurement.
#include <tiledarray.h>
#include <cstdlib>
#include <iostream>
#include <string>

#include "coo_loader.h"
#include "ta_builder.h"
#include "ta_tensors.h"
#include "ta_tensor_loader.h"
#include "t2_split.h"

extern "C" __attribute__((weak)) void MKL_Set_Num_Threads(int);

static TA::TSpArrayD permute_ij_3(const TA::TSpArrayD& src) {
  TA::TSpArrayD dst;
  dst("j,i,k") = src("i,j,k");
  return dst;
}

int main(int argc, char** argv) {
  if (MKL_Set_Num_Threads) MKL_Set_Num_Threads(1);
  TA_SCOPED_INITIALIZE(argc, argv);
  auto& world = TA::get_default_world();
  // Wait policy MUST be set AFTER init: madness::threadpool_wait_policy()
  // dereferences the ThreadPool singleton, which does not exist until
  // TA_SCOPED_INITIALIZE brings up MADNESS — calling it earlier segfaults
  // at NT>1 (this was the real cause of the "warm crash", not any TA bug).
  if (const char* v = std::getenv("SPTC_MAD_WAIT_POLICY")) {
    std::string p = v;
    int us = 0;
    if (const char* s = std::getenv("SPTC_MAD_WAIT_SLEEP_US")) us = std::atoi(s);
    if (p == "yield") madness::threadpool_wait_policy(madness::WaitPolicy::Yield);
    else if (p == "sleep") madness::threadpool_wait_policy(madness::WaitPolicy::Sleep, us);
    else if (p == "busy") madness::threadpool_wait_policy(madness::WaitPolicy::Busy);
  }
  if (argc < 2) { if (world.rank()==0) std::cerr << "usage: " << argv[0] << " <trace_dir>\n"; return 1; }

  int trials = 3, warmup = 1;
  if (const char* v = std::getenv("SPTC_TRIALS")) trials = std::atoi(v);
  if (const char* v = std::getenv("SPTC_WARMUP")) warmup = std::atoi(v) != 0;

  TATensors ts = load_ta_tensors(world, argv[1]);
  auto g_perm = permute_ij_3(ts.g);   // g_μ̃_i_Κ leaf

  run_warm_t2_bench(world, ts.c1_tot, ts.g, ts.t_i_a_tot, ts.c2_tot, ts.c2_tot,
                    ts.g0, ts.t_i_i_a_a_tot, ts.s_m_m, ts.g1, ts.f_i_m, g_perm,
                    ts.f_i_i, ts.f_m_m, trials, warmup);
  return 0;
}
