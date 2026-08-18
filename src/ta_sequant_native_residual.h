#ifndef SPTC_TA_SEQUANT_NATIVE_RESIDUAL_H
#define SPTC_TA_SEQUANT_NATIVE_RESIDUAL_H

// Thin call-site adapter mapping the native SeQuant TiledArrayGenerator's
// generated closed-shell CSV-CCSD T1/T2
// residual functions onto the existing TATensors leaf struct (populated by
// load_ta_tensors(), no new loading code needed).
//
// The generated functions' parameter lists were hand-mapped against the
// generator's own leaf_manifest() report (label + outer/inner index-space
// family signature, see tiledarray_generator.hpp), cross-referenced against
// ta_tensors.h's documented per-file leaf table:
//   native family "i"  <-> occupied            (same convention both sides)
//   native family "μ̃"  <-> loader name "m"         (PAO/uocc space)
//   native family "Κ"  <-> loader name "k"         (DF/RI auxiliary)
//   dropped-proto virtual identity "a" <-> loader name "a" (PNO/OSV virtual)
//
// Correctness note: the claim this comment used to make
// -- "passing the same underlying array under two differently-ordered
// parameters is correct, TA transposes as needed" -- is WRONG and was
// never actually verified against real execution before this note. TA's
// annotation is NOT a free relabeling per operand: `array("x,y")` binds
// array's own dim0 to label x and dim1 to label y AS-IS (no transpose
// happens just from writing labels in a different order) -- if a
// differently-ordered occurrence's generated annotation claims dim0 is the
// (114-wide) μ̃ family while the array's real dim0 is the (9-wide) i family,
// then wherever that mislabeled axis is later matched against a
// correctly-labeled μ̃ operand elsewhere in the SAME statement, TA rejects
// it with "the fused/contracted dimensions... are not congruent" -- this is
// EXACTLY what crashed both whole_t1_residual and whole_t2_residual on real
// ethane data. classify_indices() was fixed (tiledarray_generator.hpp) to
// canonicalize ToT leaves' outer order so e.g. C_ap1_μ̃ and C_μ̃_ap1 always
// agree now -- direct binding is correct for all C occurrences below. Flat
// leaves (f, g) are NOT canonicalized by that fix (classify_indices doesn't
// reorder tensors with no proto-indices at all -- SeQuant's own literal
// argument order for a flat Tensor node genuinely does vary per equation
// term, and reordering it isn't generally "more correct"). Both f_μ̃_i and
// g_μ̃_i_Κ appear with their first two outer families REVERSED relative to
// how ts.f_i_m/ts.g are physically loaded (i,μ̃,...) -- passed a genuine
// TRANSPOSED COPY below (permute_ij_2/permute_ij_3 helpers), not the raw
// array, to actually supply what those parameters' own generated code
// expects.
//
// Manifest -> TATensors field mapping (both R1 and R2 share the same 13
// distinct semantic leaves; parameter order below matches each generated
// function's own signature exactly):
//   C_ap1_μ̃, C_μ̃_ap1      -> ts.c1_tot        (C, rank-1 proto restriction)
//   C_ap2_μ̃, C_μ̃_ap2      -> ts.c2_tot        (C, rank-2 proto restriction)
//   f_i_i                  -> ts.f_i_i
//   f_i_μ̃, f_μ̃_i           -> ts.f_i_m
//   f_μ̃_μ̃                  -> ts.f_m_m
//   g_i_i_Κ                -> ts.g1
//   g_i_μ̃_Κ, g_μ̃_i_Κ       -> ts.g
//   g_μ̃_μ̃_Κ                -> ts.g0
//   s_μ̃_μ̃                  -> ts.s_m_m
//   t_ap1_i                -> ts.t_i_a_tot
//   t_ap2_ap2_i_i           -> ts.t_i_i_a_a_tot
//
// Generated via SeQuant/core/export/tiledarray_generator.hpp (mirrored at
// sptc-bench/scratch/sequant-patched) from the full cck.ipp-matching
// derivation pipeline (tests/manual/test_csv_ccsd_derivation.cpp); pasted
// here verbatim rather than re-derived at build time (SeQuant is not a
// ta-bench build dependency). Regenerate + re-paste if the derivation
// pipeline (csv_transform/optimize options, MPQC's own cck.ipp) changes.

#include "ta_tensors.h"

/// Transposed copy of a rank-2 flat array's leading two (i.e. only) axes --
/// used to correctly supply a leaf whose generated-code occurrence expects
/// the opposite outer-index order from how it was physically loaded (see
/// header comment above). A real copy, not just a differently-annotated
/// view of the same array: TA's `dst("j,i") = src("i,j")` performs the
/// permutation, not merely a relabeling.
inline TA::TSpArrayD permute_ij_2(const TA::TSpArrayD& src) {
  TA::TSpArrayD dst;
  dst("j,i") = src("i,j");
  return dst;
}

/// Same, for a rank-3 array whose first two axes need swapping while the
/// third (here, the DF/RI auxiliary "Κ" axis) stays in place.
inline TA::TSpArrayD permute_ij_3(const TA::TSpArrayD& src) {
  TA::TSpArrayD dst;
  dst("j,i,k") = src("i,j,k");
  return dst;
}

// ---- whole_t1_residual (from generated_R1.cpp, 26-term T1 residual) ------
// Parameter order matches the cache-free (`SPTC_NO_CSE=1`) generated source.
// See `tools/postprocess_generated.py --print-order` when regenerating it.
ArrayToT whole_t1_residual(
    const ArrayToT& C_ap1_uKu,
    const ArrayToT& C_uKu_ap1,
    const TA::TSpArrayD& f_i_i,
    const TA::TSpArrayD& f_i_uKu,
    const TA::TSpArrayD& f_uKu_i,
    const TA::TSpArrayD& f_uKu_uKu,
    const TA::TSpArrayD& g_i_i_K,
    const TA::TSpArrayD& g_i_uKu_K,
    const TA::TSpArrayD& g_uKu_i_K,
    const TA::TSpArrayD& g_uKu_uKu_K,
    const TA::TSpArrayD& s_uKu_uKu,
    const ArrayToT& t_ap1_i,
    const ArrayToT& t_ap2_ap2_i_i,
    const ArrayToT& C_uKu_ap2);

// ---- whole_t2_residual (from generated_R2.cpp, 55-term T2 residual) ------
// `fix_rank_collision()` in the derivation pipeline isolates any `+=` name
// collision whose expression rank changed. Parameter order matches the
// cache-free (`SPTC_NO_CSE=1`) generated source.
ArrayToT whole_t2_residual(
    const ArrayToT& C_ap2_uKu,
    const ArrayToT& C_uKu_ap1,
    const TA::TSpArrayD& f_i_i,
    const TA::TSpArrayD& f_i_uKu,
    const TA::TSpArrayD& f_uKu_uKu,
    const TA::TSpArrayD& g_i_i_K,
    const TA::TSpArrayD& g_i_uKu_K,
    const TA::TSpArrayD& g_uKu_i_K,
    const TA::TSpArrayD& g_uKu_uKu_K,
    const TA::TSpArrayD& s_uKu_uKu,
    const ArrayToT& t_ap1_i,
    const ArrayToT& t_ap2_ap2_i_i,
    const ArrayToT& C_uKu_ap2);

/// Calls whole_t1_residual() with `ts`'s fields in the mapping documented
/// above. Returns the ToT-typed T1 residual.
inline ArrayToT compute_t1_residual_native(const TATensors& ts) {
  // New cache-free order: C_ap1_μ̃, C_μ̃_ap1, f_i_i, f_i_μ̃, f_μ̃_i, f_μ̃_μ̃,
  // g_i_i_Κ, g_i_μ̃_Κ, g_μ̃_i_Κ, g_μ̃_μ̃_Κ, s_μ̃_μ̃, t_ap1_i, t_ap2_ap2_i_i, C_μ̃_ap2.
  return whole_t1_residual(ts.c1_tot, ts.c1_tot, ts.f_i_i, ts.f_i_m,
                          permute_ij_2(ts.f_i_m), ts.f_m_m, ts.g1, ts.g,
                          permute_ij_3(ts.g), ts.g0, ts.s_m_m, ts.t_i_a_tot,
                          ts.t_i_i_a_a_tot, ts.c2_tot);
}

/// Calls whole_t2_residual() with `ts`'s fields in the mapping documented
/// above. Returns the ToT-typed T2 residual.
inline ArrayToT compute_t2_residual_native(const TATensors& ts) {
  // New cache-free order: C_ap2_μ̃, C_μ̃_ap1, f_i_i, f_i_μ̃, f_μ̃_μ̃, g_i_i_Κ,
  // g_i_μ̃_Κ, g_μ̃_i_Κ, g_μ̃_μ̃_Κ, s_μ̃_μ̃, t_ap1_i, t_ap2_ap2_i_i, C_μ̃_ap2.
  return whole_t2_residual(ts.c2_tot, ts.c1_tot, ts.f_i_i, ts.f_i_m,
                          ts.f_m_m, ts.g1, ts.g, permute_ij_3(ts.g),
                          ts.g0, ts.s_m_m, ts.t_i_a_tot, ts.t_i_i_a_a_tot,
                          ts.c2_tot);
}

#endif  // SPTC_TA_SEQUANT_NATIVE_RESIDUAL_H
