#ifndef SPTC_TA_STAGE_H
#define SPTC_TA_STAGE_H

// Per-stage result type + a small utility for reading peak RSS.
// This used to live in ta_equations.h, but that header held the deprecated
// hand-coded PNO-CCSD equations and was deleted along with the codegen'd
// DLPNO set. The type stays because print_csv() and future equation
// implementations both need it.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ta_dumper.h"  // ChecksumResult, ta_compute_checksum, ta_dump_if_enabled

struct TAStageResult {
  std::string equation;
  std::string stage;
  double wall_s;
  // Time from pre-fence to immediately after `einsum()` returns. TA::einsum
  // mostly enqueues lazy task graphs and returns before remote work
  // completes, so this measures task-submission + local work only — NOT
  // end-to-end compute. `wall_s` (fence to fence) is the honest metric.
  double submit_s;
  double sparsity;
  long peak_rss_kb;
  // Correctness invariants over the LHS tensor of this stage. All four are
  // computed post-fence via ta_compute_checksum() in ta_dumper.h and
  // MPI-reduced across the world, so they are identical on every rank by
  // construction.
  int64_t nnz = 0;
  double sum = 0.0;
  double sumsq = 0.0;
  double max_abs = 0.0;
};

inline long get_peak_rss_kb() {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      long val = 0;
      std::sscanf(line.c_str(), "VmHWM: %ld kB", &val);
      return val;
    }
  }
  return 0;
}

// Shared CSV schema for all ta-bench drivers (ta_benchmark_main.cpp,
// ta_trace_benchmark_main.cpp, ta_sequant_native_residual_main.cpp) so
// results are directly comparable/concatenable across them.
inline void print_csv_header(TA::World& world) {
  if (world.rank() == 0) {
    std::cout
        << "molecule,equation,stage,trial,nranks,wall_s,submit_s,sparsity,"
        << "peak_rss_kb,nnz,sum,sumsq,max_abs,note\n";
  }
}

// Sanitize a free-form exception message so it can sit safely in a CSV cell:
// strip commas, newlines, and quotes; cap length.
inline std::string sanitize_note(const std::string& msg) {
  std::string out;
  out.reserve(msg.size());
  for (char c : msg) {
    if (c == ',' || c == '\n' || c == '\r' || c == '"') out.push_back(' ');
    else out.push_back(c);
  }
  if (out.size() > 200) out.resize(200);
  return out;
}

/// Print a set of stage results as CSV rows.
inline void print_csv(TA::World& world, const std::string& mol,
                      const std::vector<TAStageResult>& results, int trial) {
  if (world.rank() != 0) return;
  // sum/sumsq need full double precision so cross-framework verification
  // doesn't get tripped up by 6-digit default formatting noise. We restore
  // the stream's precision afterward so other CSV consumers are unaffected.
  std::ios old_state(nullptr);
  old_state.copyfmt(std::cout);
  std::cout << std::setprecision(17);
  for (const auto& r : results) {
    std::cout << mol << "," << r.equation << "," << r.stage << "," << trial
              << "," << world.size() << "," << r.wall_s << "," << r.submit_s
              << "," << r.sparsity << "," << r.peak_rss_kb
              << "," << r.nnz << "," << r.sum << "," << r.sumsq
              << "," << r.max_abs << ",\n";
  }
  std::cout.copyfmt(old_state);
  std::cout << std::flush;
}

// Emit a synthetic FAIL row when a per-equation try/catch fires so the
// surviving equations in the batch still appear cleanly in the CSV. kind is
// "OOM" or "EXC"; what is the (sanitized) exception message.
inline void print_fail_row(TA::World& world, const std::string& mol,
                           const std::string& eq, int trial,
                           const std::string& kind, const std::string& what) {
  if (world.rank() != 0) return;
  std::cout << mol << "," << eq << ",FAIL," << trial << "," << world.size()
            << ",-1,-1,0,0,0,0,0,0," << kind << ": " << sanitize_note(what)
            << "\n";
  std::cout << std::flush;
}

#endif  // SPTC_TA_STAGE_H
