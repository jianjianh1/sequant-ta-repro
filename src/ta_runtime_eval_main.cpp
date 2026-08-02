// ta_runtime_eval_main.cpp — reproduce MPQC's RUNTIME evaluator.
//
// Computes the T2 residual by walking the L2 EvalNode forest with
// sequant::evaluate() (issuing TA::einsum node-by-node + a runtime CacheManager
// for cross-term reuse), exactly as MPQC's cck.ipp does — instead of the repo's
// STATIC generated einsum sequence (src/generated_t2_residual.cpp). This is the
// definitive control artifact (predicted equal-or-slightly-worse; both paths use
// the same fenceless TA::einsum, the runtime path adds per-node lazy-cleanup
// drains). It also sidesteps the static generator's CSE deadlock bugs by using
// the runtime CacheManager (min_repeats=2) instead of static CSE.
//
// Requires SPTC_OWNING_TOT (arena inner cells dangle under the evaluator's
// cross-thread lazy cleanup). Build: -DSPTC_BUILD_RUNTIME_EVAL=ON.
//
// Derivation reproduced verbatim from
// sequant-fork/tests/manual/test_csv_ccsd_derivation.cpp (the exact provenance
// of the static file). Eval loop + leaf yielder mirror mpqc4 cck.ipp:1426-1760.

#include <tiledarray.h>

#include <SeQuant/core/context.hpp>
#include <SeQuant/core/expressions/expr_algorithms.hpp>
#include <SeQuant/core/expressions/result_expr.hpp>
#include <SeQuant/core/index_space_registry.hpp>
#include <SeQuant/core/optimize/optimize.hpp>
#include <SeQuant/core/utility/indices.hpp>
#include <SeQuant/core/utility/string.hpp>
#include <SeQuant/domain/mbpt/biorthogonalization.hpp>
#include <SeQuant/domain/mbpt/context.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>
#include <SeQuant/domain/mbpt/models/cc.hpp>
#include <SeQuant/domain/mbpt/op.hpp>
#include <SeQuant/domain/mbpt/rules/csv.hpp>
#include <SeQuant/domain/mbpt/rules/df.hpp>
#include <SeQuant/domain/mbpt/space_qns.hpp>
#include <SeQuant/domain/mbpt/utils.hpp>

#include <SeQuant/core/eval/backends/tiledarray/eval_expr.hpp>
#include <SeQuant/core/eval/backends/tiledarray/result.hpp>
#include <SeQuant/core/eval/cache_manager.hpp>
#include <SeQuant/core/eval/eval.hpp>
#include <SeQuant/core/eval/eval_node.hpp>
#include <SeQuant/core/eval/result.hpp>

#include <range/v3/view/tail.hpp>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "ta_dumper.h"
#include "ta_stage.h"
#include "ta_tensor_loader.h"
#include "ta_tensors.h"

using namespace sequant;
using EvalNodeTA = sequant::FullBinaryNode<sequant::EvalExprTA>;

// ===========================================================================
// Derivation — copied VERBATIM from test_csv_ccsd_derivation.cpp:265-346, 507
// (the exact Context + pipeline that produced src/generated_t2_residual.cpp).
// ===========================================================================
namespace {

std::shared_ptr<IndexSpaceRegistry> make_sr_spaces() {
  using namespace sequant::mbpt;
  auto isr = std::make_shared<IndexSpaceRegistry>();
  const auto spin_any = IndexSpace::QuantumNumbers{Spin::any};
  isr->add(L"i", 0b01, spin_any, is_vacuum_occupied, is_reference_occupied,
           is_hole)
      .add(L"a", 0b10, spin_any, is_particle)
      .add_union(L"p", {L"i", L"a"}, sequant::is_complete);
  mbpt::add_fermi_spin(*isr);
  mbpt::add_ao_spaces(isr, /*vbs=*/false, /*abs=*/false);
  mbpt::add_pao_spaces(isr);
  mbpt::add_df_spaces(isr);
  isr->physical_particle_attribute_mask(mask_v<Spin>);
  return isr;
}

void load_mpqc_sr_convention() {
  auto ctx = Context()
                 .set(Vacuum::SingleProduct)
                 .set(IndexSpaceMetric::Unit)
                 .set(SPBasis::Spinor)
                 .set(CanonicalizeOptions::default_options().copy_and_set(
                     CanonicalizationMethod::Complete))
                 .set(make_sr_spaces());
  set_default_context(ctx);
  mbpt::set_default_mbpt_context(
      mbpt::Context().set(mbpt::make_legacy_registry()));
}

ExprPtr tail_factor(ExprPtr const& expr) noexcept {
  if (expr->is<Tensor>())
    return expr->clone();
  else if (expr->is<Product>()) {
    auto scalar = expr->as<Product>().scalar();
    if (scalar == 1 && expr->size() == 2) return expr->at(1);
    auto facs = ranges::views::tail(*expr);
    return ex<Product>(Product{scalar, ranges::begin(facs), ranges::end(facs)});
  } else {
    auto summands = *expr | ranges::views::transform(
                                [](auto const& x) { return tail_factor(x); });
    return ex<Sum>(Sum{ranges::begin(summands), ranges::end(summands)});
  }
}

std::vector<ExprPtr> make_cceqvec_csv_closedshell(std::size_t k) {
  using sequant::mbpt::CSV;
  using sequant::mbpt::set_scoped_default_mbpt_context;
  auto biorg = [](ExprPtr const& expr, Tensor const& S) -> ExprPtr {
    auto tf = tail_factor(expr);
    auto bt = mbpt::biorthogonal_transform(tf, external_indices(S));
    bt = S.clone() * bt;
    simplify(bt);
    return bt;
  };
  auto sequant_ctx = set_scoped_default_context(
      Context(get_default_context()).set(SPBasis::Spinfree));
  auto sequant_ctx_mbpt = set_scoped_default_mbpt_context(mbpt::Context(
      {.csv = CSV::Yes, .op_registry_ptr = mbpt::make_legacy_registry()}));
  auto cc_r = mbpt::CC{k}.t();
  std::vector<ExprPtr> result;
  result.emplace_back(cc_r[0]);
  for (std::size_t r = 1; r <= k; ++r) {
    auto const S = cc_r[r]->front()->front().as<Tensor>();
    result.emplace_back(biorg(cc_r[r], S));
  }
  return result;
}

// From cck.h:507 — pins the R2 head layout to (vir<i,i>,vir<i,i>; occ, occ).
Tensor make_R_template_csv(std::size_t rank) {
  container::svector<Index> occ_idxs;
  for (std::size_t k = 1; k <= rank; ++k)
    occ_idxs.emplace_back(Index(L"i_" + std::to_wstring(k)));
  container::svector<Index> bra_idxs;
  for (std::size_t k = 1; k <= rank; ++k)
    bra_idxs.emplace_back(Index(L"a_" + std::to_wstring(k),
                                occ_idxs | ranges::to<std::vector>));
  return Tensor(L"R", bra(std::move(bra_idxs)), ket(std::move(occ_idxs)),
                Symmetry::Nonsymm, BraKetSymmetry::Nonsymm, ColumnSymmetry::Symm);
}

// Derive the optimized T2 residual Sum (mirrors test_csv_ccsd_derivation.cpp
// :450-531; opts EXACTLY as there).
ExprPtr derive_t2_residual() {
  auto residuals = make_cceqvec_csv_closedshell(2);
  auto reg = get_default_context().index_space_registry();
  auto mu = reg->retrieve(L"μ̃");
  auto Kap = reg->retrieve(L"Κ");
  ExprPtr e = residuals[2];
  e = tail_factor(e);
  e = mbpt::density_fit(e, Kap, L"g", L"g");
  e = mbpt::csv_transform(e, mu, L"C");
  flatten(e);
  OptimizeOptions opts;
  opts.opt_for = OptFor::Flops;
  opts.reorder = ReorderSum::Reorder;
  opts.is_volatile_leaf = [](Tensor const& t) { return t.label() == L"t"; };
  std::size_t proto_ext = 45;
  opts.idx_to_extent = [proto_ext](Index const& idx) -> std::size_t {
    if (idx.has_proto_indices()) return proto_ext;
    const std::wstring& key = idx.space().base_key();
    if (key == L"i") return 7;
    if (key == L"μ̃") return 114;
    if (key == L"Κ") return 282;
    return idx.space().approximate_size();
  };
  opts.n_replay = 10;
  e = optimize(e, opts);
  return e;
}

// ===========================================================================
// Leaf yielder over TATensors — mirrors cck.ipp:1426-1504 eval_csv, but serves
// the repo's already-loaded leaves and permutes each to node->annot().
// ===========================================================================

// Repo physical layouts (ta_tensors.h:30-49), as (space keys in axis order):
//   g0 (μ̃,μ̃,Κ)  g1 (i,i,Κ)  g (i,μ̃,Κ)  f_i_i (i,i)  f_i_m (i,μ̃)
//   f_m_m (μ̃,μ̃)  s_m_m (μ̃,μ̃)
// Build temp_annot = the field's physical order expressed in the node's index
// labels (grouping the node's indices by space into the physical order), then
// res(node->annot()) = field(temp_annot).
std::string space_key(const Index& ix) { return sequant::toUtf8(ix.space().base_key()); }

// Order the node's (flat) indices to match a physical space sequence.
std::string temp_annot_for(const Tensor& tnsr,
                           const std::vector<std::string>& phys_spaces) {
  std::vector<Index> idxs(tnsr.const_indices().begin(),
                          tnsr.const_indices().end());
  std::vector<bool> used(idxs.size(), false);
  std::string out;
  for (const auto& ps : phys_spaces) {
    for (std::size_t j = 0; j < idxs.size(); ++j) {
      if (!used[j] && space_key(idxs[j]) == ps) {
        if (!out.empty()) out += ",";
        out += sequant::toUtf8(idxs[j].label());
        used[j] = true;
        break;
      }
    }
  }
  return out;
}

sequant::ResultPtr yield_leaf(const EvalNodeTA& node, const TATensors& ts) {
  using ResultT = sequant::ResultTensorTA<TA::TSpArrayD>;
  using ResultToT = sequant::ResultTensorOfTensorTA<ArrayToT>;
  if (!node->is_tensor())
    return sequant::eval_result<sequant::ResultScalar<double>>(
        node->as_constant().value<double>());

  const auto& tnsr = node->as_tensor();
  const auto label = tnsr.label();

  static const bool trace = std::getenv("SPTC_EVAL_TRACE") != nullptr;
  if (trace)
    std::wcerr << L"[leaf] " << label << L"  annot='"
               << sequant::toUtf8(node->annot()).c_str() << L"' tot="
               << node->tot() << L" bra=" << tnsr.bra_rank() << L" ket="
               << tnsr.ket_rank() << L"\n" << std::flush;

  // ToT leaves: amplitudes and CSV coefficients. SPTC_CLONE_LEAF=1 returns a
  // deep clone per yield (isolating whether the evaluator aliases/consumes the
  // shared leaf array across its multiple tree positions).
  static const bool clone_leaf = std::getenv("SPTC_CLONE_LEAF") != nullptr;
  auto ret_tot = [&](const ArrayToT& a) {
    return sequant::eval_result<ResultToT>(clone_leaf ? a.clone() : a);
  };
  if (label == L"t")
    return ret_tot(tnsr.bra_rank() == 1 ? ts.t_i_a_tot : ts.t_i_i_a_a_tot);
  if (label == L"C") {
    // c1 (singles) vs c2 (doubles): the CSV coefficient carries the occupied
    // pair in the PNO index's PROTO-indices (a<i> for c1, a<i,j> for c2), NOT
    // as explicit bra/ket occ modes — so bra_rank()+ket_rank() is 2 for BOTH.
    // Discriminate by the PNO index's proto-index count (c1 -> 1, c2 -> 2).
    std::size_t max_proto = 0;
    for (const auto& ix : tnsr.const_indices())
      if (ix.has_proto_indices())
        max_proto = std::max(max_proto, ix.proto_indices().size());
    return ret_tot(max_proto >= 2 ? ts.c2_tot : ts.c1_tot);
  }

  // Flat leaves: pick field by label + space multiset, permute to node->annot().
  const TA::TSpArrayD* field = nullptr;
  std::vector<std::string> phys;  // physical space order of the field
  auto has = [&](const char* k) {
    return ranges::any_of(tnsr.const_indices(),
                          [&](const Index& ix) { return space_key(ix) == k; });
  };
  if (label == L"g") {
    if (has("i") && has("μ̃")) { field = &ts.g; phys = {"i", "μ̃", "Κ"}; }
    else if (has("i")) { field = &ts.g1; phys = {"i", "i", "Κ"}; }
    else { field = &ts.g0; phys = {"μ̃", "μ̃", "Κ"}; }
  } else if (label == L"f") {
    if (has("i") && has("μ̃")) { field = &ts.f_i_m; phys = {"i", "μ̃"}; }
    else if (has("i")) { field = &ts.f_i_i; phys = {"i", "i"}; }
    else { field = &ts.f_m_m; phys = {"μ̃", "μ̃"}; }
  } else if (label == sequant::reserved::overlap_label() || label == L"s" ||
             label == L"S") {
    field = &ts.s_m_m; phys = {"μ̃", "μ̃"};
  }
  if (!field) {
    std::wcerr << L"UNMAPPED LEAF: label='" << label << L"' annot="
               << sequant::toUtf8(node->annot()).c_str() << L"\n";
    throw std::runtime_error("yield_leaf: unmapped leaf");
  }
  const std::string temp_annot = temp_annot_for(tnsr, phys);
  TA::TSpArrayD res;
  res(node->annot()) = (*field)(temp_annot);
  return sequant::eval_result<ResultT>(std::move(res));
}

}  // namespace

int main(int argc, char** argv) {
  auto& world = TA::initialize(argc, argv);
  // SPTC_SPARSE_THRESHOLD: TA's global block-screening threshold (mirrors
  // ta_sequant_native_residual_main). =0 disables intermediate screening so the
  // result is factorization-invariant — the apples-to-apples correctness gate,
  // since the runtime and static paths screen DIFFERENT intermediates.
  if (const char* v = std::getenv("SPTC_SPARSE_THRESHOLD"))
    TA::SparseShape<float>::threshold(static_cast<float>(std::atof(v)));
  if (argc < 2) {
    if (world.rank() == 0)
      std::cerr << "usage: ta_runtime_eval_main <leaf_dir>\n";
    TA::finalize();
    return 1;
  }
  const std::string dir = argv[1];

  // 1) Derive the T2 residual forest (rank 0 derives; all ranks derive
  //    identically — SeQuant is deterministic + serial here).
  load_mpqc_sr_convention();
  ExprPtr e = derive_t2_residual();
  if (world.rank() == 0) {
    std::cerr << "derived T2 residual: "
              << (e->is<Sum>() ? e->as<Sum>().size() : 1) << " summands\n";
  }

  auto tmpl = make_R_template_csv(2);
  std::string annot = sequant::csv_labels(tmpl.ket()) + ";" +
                      sequant::csv_labels(tmpl.bra());

  std::vector<EvalNodeTA> nodes;
  if (e->is<Sum>()) {
    for (const auto& summand : e->as<Sum>().summands())
      nodes.push_back(sequant::binarize<sequant::EvalExprTA>(
          sequant::ResultExpr{tmpl, summand}));
  } else {
    nodes.push_back(
        sequant::binarize<sequant::EvalExprTA>(sequant::ResultExpr{tmpl, e}));
  }

  auto is_t_leaf = [](EvalNodeTA const& n) {
    return n.leaf() && n->is_tensor() && n->as_tensor().label() == L"t";
  };
  // SPTC_NO_CACHE=1 disables intermediate caching (huge min_repeats) to isolate
  // whether the CacheManager is behind the owning-ToT heap corruption.
  auto cache = std::getenv("SPTC_NO_CACHE")
                   ? sequant::cache_manager(nodes, std::size_t(1) << 40)
                   : sequant::cache_manager(nodes, is_t_leaf, /*min_repeats=*/2);

  if (std::getenv("SPTC_EVAL_TRACE")) {
    std::function<void(const EvalNodeTA&, int)> dump =
        [&](const EvalNodeTA& n, int d) {
          std::wcerr << std::wstring(2 * d, L' ');
          if (n.leaf())
            std::wcerr << L"LEAF "
                       << (n->is_tensor() ? n->as_tensor().label()
                                          : std::wstring(L"<const>"));
          else
            std::wcerr << L"NODE op="
                       << (n->op_type() ? static_cast<int>(*n->op_type()) : -1);
          std::wcerr << L"  annot='" << sequant::toUtf8(n->annot()).c_str()
                     << L"' tot=" << n->tot() << L"\n";
          if (!n.leaf()) { dump(n.left(), d + 1); dump(n.right(), d + 1); }
        };
    std::wcerr << L"===== summand 0 tree =====\n";
    dump(nodes.front(), 0);
    std::wcerr << std::flush;
  }

  // 2) Load leaves.
  TATensors ts = load_ta_tensors(world, dir);
  auto leaf_eval = [&ts](EvalNodeTA const& n) { return yield_leaf(n, ts); };

  if (std::getenv("SPTC_EVAL_TRACE") && world.rank() == 0) {
    std::cerr << "g0 trange:  " << ts.g0.trange() << "\n";
    std::cerr << "g  trange:  " << ts.g.trange() << "\n";
    std::cerr << "c2_tot trange: " << ts.c2_tot.trange() << "\n";
    std::cerr << "c1_tot trange: " << ts.c1_tot.trange() << "\n" << std::flush;
  }

  // SPTC_CHECK_SYM: is the ToT coeff/amplitude symmetric under the occ pair
  // swap i<->j? If ‖X(i,j;·) - X(j,i;·)‖ ~ 0 the yielder's return-as-is is
  // exact; if large, the swapped-occ annotation the derivation applies to some
  // occurrences is the sub-% error source.
  if (std::getenv("SPTC_CHECK_SYM")) {
    world.gop.fence();
    ArrayToT c_sym;
    c_sym("i,j,x;a") = ts.c2_tot("i,j,x;a") - ts.c2_tot("j,i,x;a");
    ArrayToT t_sym;
    t_sym("i,j;a,b") = ts.t_i_i_a_a_tot("i,j;a,b") - ts.t_i_i_a_a_tot("j,i;b,a");
    world.gop.fence();
    auto cc = ta_compute_checksum(world, c_sym);
    auto ct = ta_compute_checksum(world, t_sym);
    if (world.rank() == 0) {
      std::cout << "  [sym] c2_tot i<->j asym: sumsq=" << std::setprecision(6)
                << cc.sumsq << " max_abs=" << cc.max_abs << " (vs c2 sumsq "
                << "below)\n";
      std::cout << "  [sym] t2 (ij,ab)<->(ji,ba) asym: sumsq=" << ct.sumsq
                << " max_abs=" << ct.max_abs << "\n" << std::flush;
    }
    auto cref = ta_compute_checksum(world, ts.c2_tot);
    auto tref = ta_compute_checksum(world, ts.t_i_i_a_a_tot);
    if (world.rank() == 0)
      std::cout << "  [sym] ref: c2 sumsq=" << cref.sumsq << " t2 sumsq="
                << tref.sumsq << "\n" << std::flush;
  }

  // Minimal reproducer for the flat(μ̃,μ̃)×ToT-C2 shape-gemm crash: contract a
  // flat (μ̃,μ̃) [f_m_m, same 8×8 μ̃ tiling as g0] against c2_tot several ways.
  if (const char* v = std::getenv("SPTC_REPRO_EINSUM")) {
    // SPTC_REPRO_NOFENCE=1 removes the inter-step fences so the chain runs
    // exactly as the evaluator drives it (lazy futures, no barrier between
    // einsums) — tests the missing-dependency hypothesis.
    const bool no_fence = std::getenv("SPTC_REPRO_NOFENCE") != nullptr;
    auto trial = [&](const char* tag, auto&& fn) {
      if (!no_fence) world.gop.fence();
      try {
        fn();
        if (!no_fence) world.gop.fence();
        std::cerr << "  REPRO[" << tag << "]: OK\n" << std::flush;
      } catch (const std::exception& e) {
        std::cerr << "  REPRO[" << tag << "]: EXC " << e.what() << "\n"
                  << std::flush;
      }
    };
    // Replay summand-0's exact 4-step chain (from the trace) with real leaves,
    // fencing + catching after each step to pinpoint the corrupting einsum.
    // labels: _a,_b = μ̃ ; _i,_j = occ ; _k = Κ ; _p = inner PNO a
    TA::TSpArrayD I1, I2, I3;
    ArrayToT I4;
    trial("s1_CxT_denest", [&] {  // C1(i,μ̃;a) × t1(i;a) -> (μ̃,i)  [ToT×ToT→flat]
      I1 = TA::einsum<TA::DeNest::True>(ts.c1_tot("_i,_a;_p"),
                                        ts.t_i_a_tot("_i;_p"), "_a,_i");
    });
    trial("s2_x_g", [&] {  // (μ̃,i) × g(i,μ̃,Κ) -> (Κ)  [evaluator order: g LEFT]
      I2 = TA::einsum(ts.g("_i,_a,_k"), I1("_a,_i"), "_k");
    });
    trial("s3_x_g0", [&] {  // (Κ) × g0(μ̃,μ̃,Κ) -> (μ̃,μ̃)
      I3 = TA::einsum(I2("_k"), ts.g0("_a,_b,_k"), "_a,_b");
    });
    trial("s4_x_C2", [&] {  // (μ̃,μ̃) × C2(i,i,μ̃;a) -> (i,i,μ̃;a)  [flat×ToT→ToT]
      I4 = TA::einsum(I3("_a,_b"), ts.c2_tot("_i,_j,_a;_p"), "_i,_j,_b;_p");
    });
    // s4b: EXACT evaluator combo — derived I3 + occ SWAPPED (i_2,i_1 vs build
    // order i_1,i_2) + pair-subscripted inner label.
    ArrayToT I4b;
    trial("s4b_swapocc", [&] {
      I4b = TA::einsum(I3("m1,m2"), ts.c2_tot("i2,i1,m1;a2i1i2"),
                       "i2,i1,m2;a2i1i2");
    });
    // s5: ToT×ToT→ToT keeping BOTH inner PNOs (double inner), contract μ̃.
    ArrayToT I5;
    trial("s5_TxT_double_inner", [&] {
      I5 = TA::einsum(I4b("i2,i1,m2;a2i1i2"), ts.c2_tot("i2,i1,m2;a4i1i2"),
                      "i2,i1;a2i1i2,a4i1i2");
    });
    // s6: × t2(i,i;a,a) contract one inner a_4 -> (i,i;a2,a1)
    ArrayToT I6;
    trial("s6_x_t2", [&] {
      I6 = TA::einsum(I5("i2,i1;a2i1i2,a4i1i2"),
                      ts.t_i_i_a_a_tot("i2,i1;a4i1i2,a1i1i2"),
                      "i1,i2;a2i1i2,a1i1i2");
    });
    TA::finalize();
    return 0;
  }

  // 3) Evaluate (cold: no reuse across trials for now) + symmetrize + checksum.
  const int trials = [] {
    const char* v = std::getenv("SPTC_TRIALS");
    return v ? std::max(1, std::atoi(v)) : 1;
  }();
  for (int trial = 0; trial < trials; ++trial) {
    world.gop.fence();
    auto t0 = std::chrono::high_resolution_clock::now();
    ArrayToT R2 =
        sequant::evaluate(nodes.front(), annot, leaf_eval, cache)
            ->template get<ArrayToT>();
    for (auto&& n : ranges::views::tail(nodes)) {
      auto temp = sequant::evaluate(n, annot, leaf_eval, cache)
                      ->template get<ArrayToT>();
      R2(annot) += temp(annot);
    }
    auto t_submit = std::chrono::high_resolution_clock::now();
    world.gop.fence();
    {
      auto pre = ta_compute_checksum(world, R2);
      if (world.rank() == 0)
        std::cout << "  [pre-symm] nnz=" << pre.nnz << " sum="
                  << std::setprecision(15) << pre.sum << " sumsq=" << pre.sumsq
                  << " max_abs=" << pre.max_abs << "\n" << std::flush;
    }
    // R2 pair symmetrization (cck.ipp:1754); SPTC_NO_SYMM=1 skips it (to A/B
    // against the static path, whose symmetrization is SPTC_SYMMETRIZE_R2-gated).
    if (!std::getenv("SPTC_NO_SYMM"))
      R2("i,j;a,b") = 0.5 * (R2("i,j;a,b") + R2("j,i;b,a"));
    world.gop.fence();
    auto t_wall = std::chrono::high_resolution_clock::now();

    auto cs = ta_compute_checksum(world, R2);
    double submit_s =
        std::chrono::duration<double>(t_submit - t0).count();
    double wall_s = std::chrono::duration<double>(t_wall - t0).count();
    if (world.rank() == 0) {
      std::cout << "runtime_t2_residual residual checksum: nnz=" << cs.nnz
                << " sum=" << std::setprecision(15) << cs.sum
                << " sumsq=" << cs.sumsq << " max_abs=" << cs.max_abs
                << " wall_s=" << wall_s << " submit_s=" << submit_s << "\n"
                << std::flush;
    }
  }

  TA::finalize();
  return 0;
}
