// Per-operation trace wrapper for the performance-parity investigation
// (Phase A: track the TiledArray operation-stream difference vs real MPQC).
//
// `sptc::einsum<DeNest>(a, b, out)` is a drop-in for `TA::einsum<DeNest>(a,
// b, out)`: it fences, times just this contraction, computes the same
// 4-tuple checksum MPQC's SEQUANT_EVAL_TRACE emits per node (reusing
// ta_dumper.h's ta_compute_checksum), logs one CSV row, then returns
// exactly what TA::einsum returns (so the generated `(...)("out")`
// re-annotation still works). Enable by compiling the traced copies of
// generated_t{1,2}_residual.cpp against this header; output path via
// SPTC_TRACE_OPS_PATH (default ./sptc_ops_trace.csv). Fencing per op
// serializes execution, so wall_ns here is per-op cost, NOT the tuned
// end-to-end number — this build is for the op-stream diff, not timing.
#ifndef SPTC_TRACED_EINSUM_H
#define SPTC_TRACED_EINSUM_H

#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>

#include "ta_dumper.h"  // ta_compute_checksum(world, {TSpArrayD|ArrayToT})

namespace sptc {

inline int& op_counter() {
  static int c = 0;
  return c;
}

inline std::ofstream& trace_out() {
  static std::ofstream f([] {
    const char* p = std::getenv("SPTC_TRACE_OPS_PATH");
    return std::string(p ? p : "sptc_ops_trace.csv");
  }());
  static bool header = [&] {
    f << "op,out_annot,result_volume,nnz,sum,sumsq,max_abs,wall_ns\n";
    return true;
  }();
  (void)header;
  return f;
}

// Drop-in for TA::einsum<DeNest>(a, b, out): forward, fence, checksum, log.
template <TA::DeNest DeNestV = TA::DeNest::False, typename ExprA,
          typename ExprB>
auto einsum(const ExprA& a, const ExprB& b, const std::string& out) {
  auto& w = TA::get_default_world();
  w.gop.fence();
  const auto t0 = std::chrono::high_resolution_clock::now();
  auto result = TA::einsum<DeNestV>(a, b, out);
  w.gop.fence();
  const auto t1 = std::chrono::high_resolution_clock::now();
  const long ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  const auto cs = ta_compute_checksum(w, result);
  const std::size_t vol = result.trange().elements_range().volume();
  if (w.rank() == 0) {
    trace_out() << (++op_counter()) << ",\"" << out << "\"," << vol << ","
                << cs.nnz << "," << cs.sum << "," << cs.sumsq << ","
                << cs.max_abs << "," << ns << "\n";
    trace_out().flush();
  }
  return result;
}

}  // namespace sptc

#endif  // SPTC_TRACED_EINSUM_H
