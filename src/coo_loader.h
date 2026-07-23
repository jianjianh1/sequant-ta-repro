#ifndef SPTC_COO_LOADER_H
#define SPTC_COO_LOADER_H

#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/// Sparse tensor in COO (coordinate) format, up to rank 4.
struct COOTensor {
  int rank = 0;
  std::vector<std::array<std::size_t, 4>> indices;
  std::vector<double> values;
  std::array<std::size_t, 4> shape = {0, 0, 0, 0};  // max_index + 1 per dim
};

/// Load a sparse tensor from a space/tab-separated text file.
/// Each line: idx0 idx1 ... idxN value
/// Rank is auto-detected from the first non-empty line.
inline COOTensor load_coo(const std::string& filename) {
  COOTensor tensor;
  std::ifstream in(filename);
  if (!in.is_open()) {
    std::cerr << "ERROR: cannot open " << filename << "\n";
    return tensor;
  }

  std::string line;
  bool first = true;
  // Fallback shape/rank parsed from the `# shape=... rank=...` header —
  // needed when a tensor has zero stored elements (nnz=0, e.g. ethane's
  // f_i_1_m_1.tns): with no data rows, rank/shape can't be inferred from
  // the data scan below, and without this fallback the tensor would come
  // back as an invalid rank=0/all-zero-shape COOTensor.
  int header_rank = 0;
  std::array<std::size_t, 4> header_shape = {0, 0, 0, 0};

  while (std::getline(in, line)) {
    if (line.empty()) continue;
    // Skip `#`-prefixed comment/metadata lines (whitespace-tolerant).
    // MPQC-derived .tns files carry a `# shape=... nnz=... sum=...`
    // header on line 1; without this skip the loader interprets the
    // 9 metadata tokens as an 8-index tensor row.
    {
      auto p = line.find_first_not_of(" \t");
      if (p != std::string::npos && line[p] == '#') {
        auto shape_pos = line.find("shape=");
        auto rank_pos = line.find("rank=");
        if (shape_pos != std::string::npos) {
          std::string rest = line.substr(shape_pos + 6);
          std::istringstream ss(rest.substr(0, rest.find(' ')));
          std::string tok;
          int d = 0;
          while (d < 4 && std::getline(ss, tok, ','))
            header_shape[d++] = std::stoull(tok);
        }
        if (rank_pos != std::string::npos)
          header_rank = std::stoi(line.substr(rank_pos + 5));
        continue;
      }
    }
    std::istringstream iss(line);

    // On first line, detect rank by counting tokens
    if (first) {
      std::vector<std::string> tokens;
      std::string tok;
      while (iss >> tok) tokens.push_back(tok);
      tensor.rank = static_cast<int>(tokens.size()) - 1;  // last token is value
      if (tensor.rank < 1 || tensor.rank > 4) {
        std::cerr << "ERROR: unsupported rank " << tensor.rank << " in "
                  << filename << "\n";
        return tensor;
      }

      std::array<std::size_t, 4> idx = {0, 0, 0, 0};
      for (int d = 0; d < tensor.rank; ++d) {
        idx[d] = std::stoull(tokens[d]);
        if (idx[d] + 1 > tensor.shape[d]) tensor.shape[d] = idx[d] + 1;
      }
      double val = std::stod(tokens[tensor.rank]);
      tensor.indices.push_back(idx);
      tensor.values.push_back(val);
      first = false;
      continue;
    }

    std::array<std::size_t, 4> idx = {0, 0, 0, 0};
    for (int d = 0; d < tensor.rank; ++d) {
      iss >> idx[d];
      if (idx[d] + 1 > tensor.shape[d]) tensor.shape[d] = idx[d] + 1;
    }
    double val;
    iss >> val;
    tensor.indices.push_back(idx);
    tensor.values.push_back(val);
  }

  // Zero data rows (nnz=0): no rank/shape could be inferred from the scan
  // above. Fall back to the header's declared shape/rank so the tensor is
  // still a valid (all-zero) array of the right dimensionality, rather than
  // an invalid rank=0 COOTensor.
  if (tensor.rank == 0 && header_rank > 0) {
    tensor.rank = header_rank;
    tensor.shape = header_shape;
  }

  return tensor;
}

#endif  // SPTC_COO_LOADER_H
