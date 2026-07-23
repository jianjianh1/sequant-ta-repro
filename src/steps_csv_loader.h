#ifndef SPTC_STEPS_CSV_LOADER_H
#define SPTC_STEPS_CSV_LOADER_H

// Parses an MPQC eval-trace steps.csv (see mpqc4:traces/checksum-run/*.steps.csv)
// into per-(iter,term_idx) ordered lists of real "Product" step checksums —
// the 1:1 join key against mpqc_trace_equations.h's generated stage order.
// extract_trace_equations.py's own stage numbering already matches this real
// per-term Product/residual step sequence (see that script's docstring and
// its 815/815 node-level cross-check against MPQC's own logged targets).

#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct RealStageChecksum {
  int64_t nnz = 0;
  double sum = 0.0;
  double sumsq = 0.0;
  double max_abs = 0.0;
  std::string target_expr;  // for diagnostics only
};

namespace detail {

// Minimal RFC4180 CSV line splitter: handles double-quoted fields with
// embedded commas and "" escaped quotes. Good enough for this one file's
// shape (no embedded newlines inside a field).
inline std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::string cur;
  bool in_quotes = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    char c = line[i];
    if (in_quotes) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') {
          cur.push_back('"');
          ++i;
        } else {
          in_quotes = false;
        }
      } else {
        cur.push_back(c);
      }
    } else {
      if (c == '"') {
        in_quotes = true;
      } else if (c == ',') {
        fields.push_back(cur);
        cur.clear();
      } else {
        cur.push_back(c);
      }
    }
  }
  fields.push_back(cur);
  return fields;
}

}  // namespace detail

// block_id -> ordered list of real checksums, one per real "Product" step
// (step_kind == "Product", covers both intermediate results and the final
// residual/scale stage) in the same order extract_trace_equations.py's
// stages appear — i.e. 1:1 with mpqc_trace_equations.h's ta_run_* stage
// vector. "Permute"/"Tensor"(leaf)/"Constant" rows are real trace lines too
// but don't correspond to a DSL stage (Permute re-states an already-computed
// residual; Tensor/Constant are operand references, not new results) so
// they're intentionally excluded from the join.
inline std::map<std::string, std::vector<RealStageChecksum>> load_real_checksums(
    const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open steps.csv: " + path);

  std::string header_line;
  if (!std::getline(f, header_line))
    throw std::runtime_error("empty steps.csv: " + path);
  auto header = detail::split_csv_line(header_line);
  std::map<std::string, std::size_t> col;
  for (std::size_t i = 0; i < header.size(); ++i) col[header[i]] = i;
  for (const char* required : {"iter", "term_idx", "step_kind", "checksum_nnz",
                               "checksum_sum", "checksum_sumsq",
                               "checksum_max_abs", "target_expr"}) {
    if (!col.count(required))
      throw std::runtime_error(std::string("steps.csv missing column: ") + required);
  }

  std::map<std::string, std::vector<RealStageChecksum>> out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    auto fields = detail::split_csv_line(line);
    if (fields.size() <= col.at("target_expr")) continue;
    const std::string& step_kind = fields[col.at("step_kind")];
    if (step_kind != "Product") continue;  // see doc comment above
    // "Product scalar" rows (kind=="scalar", e.g. "Z * 2 -> E") are real
    // trace lines for energy contributions, which extract_trace_equations.py
    // deliberately excludes from its residual catalog (see that script's
    // docstring) — they have no tensor checksum (nnz/sum/etc. fields are
    // blank), and their term_idx never appears in the generated registry.
    // Skip rather than crash on the empty numeric field.
    if (fields[col.at("checksum_nnz")].empty()) continue;

    std::string iter = fields[col.at("iter")];
    std::string term_idx = fields[col.at("term_idx")];
    std::string block_id = "iter" + iter + "_term" + term_idx;

    RealStageChecksum cs;
    cs.nnz = std::stoll(fields[col.at("checksum_nnz")]);
    cs.sum = std::stod(fields[col.at("checksum_sum")]);
    cs.sumsq = std::stod(fields[col.at("checksum_sumsq")]);
    cs.max_abs = std::stod(fields[col.at("checksum_max_abs")]);
    cs.target_expr = fields[col.at("target_expr")];
    out[block_id].push_back(cs);
  }
  return out;
}

#endif  // SPTC_STEPS_CSV_LOADER_H
