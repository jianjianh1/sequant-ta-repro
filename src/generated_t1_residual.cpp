// AUTO-GENERATED (2026-08-03) by SeQuant's native TiledArrayGenerator
// (SeQuant/core/export/tiledarray_generator.hpp) from the full
// cck.ipp-matching closed-shell CSV-CCSD T1 residual derivation
// (tests/manual/test_csv_ccsd_derivation.cpp), emitted CACHE-FREE with
// SPTC_NO_CSE=1: the forest is exported un-deduped, per-summand, with NO
// cross-term common-subexpression elimination -- i.e. no reuse of shared
// intermediates across terms. This is the "without cache/reuse" contraction
// sequence the benchmark drives (README.md; docs/HARNESS_VS_EXPERIMENTS.md).
// The arena ToT type the generator hardcodes is rewritten to the ArrayToT
// alias (ta_tensors.h) so this one source builds both the owning and arena
// backends. Regenerate + re-run tools/postprocess_generated.py (do not
// hand-edit); re-sync src/ta_sequant_native_residual.h if the parameter order
// changed (see --print-order).
#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>
#include <cmath>
#include "ta_tensors.h"

ArrayToT whole_t1_residual(const ArrayToT& C_ap1_μ̃, const ArrayToT& C_μ̃_ap1, const TA::TSpArrayD& f_i_i, const TA::TSpArrayD& f_i_μ̃, const TA::TSpArrayD& f_μ̃_i, const TA::TSpArrayD& f_μ̃_μ̃, const TA::TSpArrayD& g_i_i_Κ, const TA::TSpArrayD& g_i_μ̃_Κ, const TA::TSpArrayD& g_μ̃_i_Κ, const TA::TSpArrayD& g_μ̃_μ̃_Κ, const TA::TSpArrayD& s_μ̃_μ̃, const ArrayToT& t_ap1_i, const ArrayToT& t_ap2_ap2_i_i, const ArrayToT& C_μ̃_ap2) {
  TA::TSpArrayD I_Κ;
  TA::TSpArrayD I_i_i;
  ArrayToT I_i_ap1;
  TA::TSpArrayD I_i_μ̃;
  TA::TSpArrayD I_μ̃_μ̃;
  TA::TSpArrayD I_i_i_Κ;
  ArrayToT I_i_ap2_Κ;
  TA::TSpArrayD I_i_μ̃_Κ;
  ArrayToT I_i_i_i_ap2;
  TA::TSpArrayD I_i_i_i_μ̃;
  ArrayToT I_i_i_ap2_ap2;
  ArrayToT I_i_i_ap2_μ̃;
  TA::TSpArrayD I_i_i_μ̃_μ̃;
  ArrayToT I2_i_ap2;
  TA::TSpArrayD I2_i_μ̃;
  ArrayToT I3_i_ap2;
  TA::TSpArrayD I3_i_μ̃;
  TA::TSpArrayD I4_i_μ̃;
  ArrayToT I2_i_ap1;
  I3_i_μ̃("i_2,μ̃_19580") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19580;a_2"), t_ap1_i("i_2;a_2"), "i_2,μ̃_19580")("i_2,μ̃_19580");
  I2_i_μ̃("i_1,μ̃_19580") = TA::einsum(I3_i_μ̃("i_2,μ̃_19580"), f_i_i("i_2,i_1"), "i_1,μ̃_19580")("i_1,μ̃_19580");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_1,μ̃_19579") = TA::einsum(I2_i_μ̃("i_1,μ̃_19580"), s_μ̃_μ̃("μ̃_19579,μ̃_19580"), "i_1,μ̃_19579")("i_1,μ̃_19579");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") = (TA::einsum(I_i_μ̃("i_1,μ̃_19579"), C_ap1_μ̃("i_1,μ̃_19579;a_1"), "i_1;a_1")("i_1;a_1")) * (-1);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_μ̃("i_2,μ̃_19581") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19581;a_2"), t_ap1_i("i_2;a_2"), "i_2,μ̃_19581")("i_2,μ̃_19581");
  I_Κ("Κ_1") = TA::einsum(I2_i_μ̃("i_2,μ̃_19581"), g_i_μ̃_Κ("i_2,μ̃_19581,Κ_1"), "Κ_1")("Κ_1");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_1,μ̃_19582") = TA::einsum(I_Κ("Κ_1"), g_μ̃_i_Κ("μ̃_19582,i_1,Κ_1"), "i_1,μ̃_19582")("i_1,μ̃_19582");
  I_Κ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19582"), C_ap1_μ̃("i_1,μ̃_19582;a_1"), "i_1;a_1")("i_1;a_1")) * (2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_μ̃("i_2,μ̃_19586") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19586;a_2"), t_ap1_i("i_2;a_2"), "i_2,μ̃_19586")("i_2,μ̃_19586");
  I_Κ("Κ_1") = TA::einsum(I2_i_μ̃("i_2,μ̃_19586"), g_i_μ̃_Κ("i_2,μ̃_19586,Κ_1"), "Κ_1")("Κ_1");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_μ̃_μ̃("μ̃_19587,μ̃_19588") = TA::einsum(I_Κ("Κ_1"), g_μ̃_μ̃_Κ("μ̃_19587,μ̃_19588,Κ_1"), "μ̃_19587,μ̃_19588")("μ̃_19587,μ̃_19588");
  I_Κ = TA::TSpArrayD();  // release
  I2_i_μ̃("i_1,μ̃_19588") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_1,μ̃_19588;a_3"), t_ap1_i("i_1;a_3"), "i_1,μ̃_19588")("i_1,μ̃_19588");
  I_i_μ̃("i_1,μ̃_19587") = TA::einsum(I_μ̃_μ̃("μ̃_19587,μ̃_19588"), I2_i_μ̃("i_1,μ̃_19588"), "i_1,μ̃_19587")("i_1,μ̃_19587");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19587"), C_ap1_μ̃("i_1,μ̃_19587;a_1"), "i_1;a_1")("i_1;a_1")) * (2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_μ̃("i_3,μ̃_19623") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_3,μ̃_19623;a_2"), t_ap1_i("i_3;a_2"), "i_3,μ̃_19623")("i_3,μ̃_19623");
  I_Κ("Κ_1") = TA::einsum(I2_i_μ̃("i_3,μ̃_19623"), g_i_μ̃_Κ("i_3,μ̃_19623,Κ_1"), "Κ_1")("Κ_1");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i("i_1,i_2") = TA::einsum(I_Κ("Κ_1"), g_i_i_Κ("i_2,i_1,Κ_1"), "i_1,i_2")("i_1,i_2");
  I_Κ = TA::TSpArrayD();  // release
  I3_i_μ̃("i_2,μ̃_19625") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19625;a_3"), t_ap1_i("i_2;a_3"), "i_2,μ̃_19625")("i_2,μ̃_19625");
  I2_i_μ̃("i_2,μ̃_19624") = TA::einsum(I3_i_μ̃("i_2,μ̃_19625"), s_μ̃_μ̃("μ̃_19624,μ̃_19625"), "i_2,μ̃_19624")("i_2,μ̃_19624");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_1,μ̃_19624") = TA::einsum(I_i_i("i_1,i_2"), I2_i_μ̃("i_2,μ̃_19624"), "i_1,μ̃_19624")("i_1,μ̃_19624");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19624"), C_ap1_μ̃("i_1,μ̃_19624;a_1"), "i_1;a_1")("i_1;a_1")) * (-2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I4_i_μ̃("i_2,μ̃_19626") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19626;a_2"), t_ap1_i("i_2;a_2"), "i_2,μ̃_19626")("i_2,μ̃_19626");
  I_Κ("Κ_1") = TA::einsum(I4_i_μ̃("i_2,μ̃_19626"), g_i_μ̃_Κ("i_2,μ̃_19626,Κ_1"), "Κ_1")("Κ_1");
  I4_i_μ̃ = TA::TSpArrayD();  // release
  I3_i_μ̃("i_3,μ̃_19627") = TA::einsum(I_Κ("Κ_1"), g_i_μ̃_Κ("i_3,μ̃_19627,Κ_1"), "i_3,μ̃_19627")("i_3,μ̃_19627");
  I_Κ = TA::TSpArrayD();  // release
  I3_i_ap2("i_1,i_3;a_3") = TA::einsum(I3_i_μ̃("i_3,μ̃_19627"), C_μ̃_ap2("i_1,i_3,μ̃_19627;a_3"), "i_1,i_3;a_3")("i_1,i_3;a_3");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_ap2("i_3,i_1;a_4") = TA::einsum(I3_i_ap2("i_1,i_3;a_3"), t_ap2_ap2_i_i("i_3,i_1;a_3,a_4"), "i_3,i_1;a_4")("i_3,i_1;a_4");
  I3_i_ap2 = ArrayToT();  // release
  I2_i_μ̃("i_1,μ̃_19629") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_3,i_1;a_4"), C_μ̃_ap2("i_1,i_3,μ̃_19629;a_4"), "i_1,μ̃_19629")("i_1,μ̃_19629");
  I2_i_ap2 = ArrayToT();  // release
  I_i_μ̃("i_1,μ̃_19628") = TA::einsum(I2_i_μ̃("i_1,μ̃_19629"), s_μ̃_μ̃("μ̃_19628,μ̃_19629"), "i_1,μ̃_19628")("i_1,μ̃_19628");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19628"), C_ap1_μ̃("i_1,μ̃_19628;a_1"), "i_1;a_1")("i_1;a_1")) * (4);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I3_i_μ̃("i_2,μ̃_19642") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19642;a_2"), t_ap1_i("i_2;a_2"), "i_2,μ̃_19642")("i_2,μ̃_19642");
  I_Κ("Κ_1") = TA::einsum(I3_i_μ̃("i_2,μ̃_19642"), g_i_μ̃_Κ("i_2,μ̃_19642,Κ_1"), "Κ_1")("Κ_1");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_μ̃("i_3,μ̃_19643") = TA::einsum(I_Κ("Κ_1"), g_i_μ̃_Κ("i_3,μ̃_19643,Κ_1"), "i_3,μ̃_19643")("i_3,μ̃_19643");
  I_Κ = TA::TSpArrayD();  // release
  I3_i_μ̃("i_1,μ̃_19643") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_1,μ̃_19643;a_3"), t_ap1_i("i_1;a_3"), "i_1,μ̃_19643")("i_1,μ̃_19643");
  I_i_i("i_1,i_3") = TA::einsum(I2_i_μ̃("i_3,μ̃_19643"), I3_i_μ̃("i_1,μ̃_19643"), "i_1,i_3")("i_1,i_3");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I3_i_μ̃("i_3,μ̃_19645") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_3,μ̃_19645;a_4"), t_ap1_i("i_3;a_4"), "i_3,μ̃_19645")("i_3,μ̃_19645");
  I2_i_μ̃("i_3,μ̃_19644") = TA::einsum(I3_i_μ̃("i_3,μ̃_19645"), s_μ̃_μ̃("μ̃_19644,μ̃_19645"), "i_3,μ̃_19644")("i_3,μ̃_19644");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_1,μ̃_19644") = TA::einsum(I_i_i("i_1,i_3"), I2_i_μ̃("i_3,μ̃_19644"), "i_1,μ̃_19644")("i_1,μ̃_19644");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19644"), C_ap1_μ̃("i_1,μ̃_19644;a_1"), "i_1;a_1")("i_1;a_1")) * (-2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I4_i_μ̃("i_2,μ̃_19653") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19653;a_2"), t_ap1_i("i_2;a_2"), "i_2,μ̃_19653")("i_2,μ̃_19653");
  I_Κ("Κ_1") = TA::einsum(I4_i_μ̃("i_2,μ̃_19653"), g_i_μ̃_Κ("i_2,μ̃_19653,Κ_1"), "Κ_1")("Κ_1");
  I4_i_μ̃ = TA::TSpArrayD();  // release
  I3_i_μ̃("i_3,μ̃_19654") = TA::einsum(I_Κ("Κ_1"), g_i_μ̃_Κ("i_3,μ̃_19654,Κ_1"), "i_3,μ̃_19654")("i_3,μ̃_19654");
  I_Κ = TA::TSpArrayD();  // release
  I3_i_ap2("i_1,i_3;a_3") = TA::einsum(I3_i_μ̃("i_3,μ̃_19654"), C_μ̃_ap2("i_1,i_3,μ̃_19654;a_3"), "i_1,i_3;a_3")("i_1,i_3;a_3");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_ap2("i_3,i_1;a_4") = TA::einsum(I3_i_ap2("i_1,i_3;a_3"), t_ap2_ap2_i_i("i_1,i_3;a_3,a_4"), "i_3,i_1;a_4")("i_3,i_1;a_4");
  I3_i_ap2 = ArrayToT();  // release
  I2_i_μ̃("i_1,μ̃_19656") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_3,i_1;a_4"), C_μ̃_ap2("i_1,i_3,μ̃_19656;a_4"), "i_1,μ̃_19656")("i_1,μ̃_19656");
  I2_i_ap2 = ArrayToT();  // release
  I_i_μ̃("i_1,μ̃_19655") = TA::einsum(I2_i_μ̃("i_1,μ̃_19656"), s_μ̃_μ̃("μ̃_19655,μ̃_19656"), "i_1,μ̃_19655")("i_1,μ̃_19655");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19655"), C_ap1_μ̃("i_1,μ̃_19655;a_1"), "i_1;a_1")("i_1;a_1")) * (-2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_μ̃("i_1,μ̃_19583") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_1,μ̃_19583;a_2"), t_ap1_i("i_1;a_2"), "i_1,μ̃_19583")("i_1,μ̃_19583");
  I_i_i_Κ("i_1,i_2,Κ_1") = TA::einsum(I2_i_μ̃("i_1,μ̃_19583"), g_i_μ̃_Κ("i_2,μ̃_19583,Κ_1"), "i_1,i_2,Κ_1")("i_1,i_2,Κ_1");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_μ̃("i_2,μ̃_19585") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19585;a_3"), t_ap1_i("i_2;a_3"), "i_2,μ̃_19585")("i_2,μ̃_19585");
  I_i_μ̃_Κ("i_2,μ̃_19584,Κ_1") = TA::einsum(I2_i_μ̃("i_2,μ̃_19585"), g_μ̃_μ̃_Κ("μ̃_19584,μ̃_19585,Κ_1"), "i_2,μ̃_19584,Κ_1")("i_2,μ̃_19584,Κ_1");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_1,μ̃_19584") = TA::einsum(I_i_i_Κ("i_1,i_2,Κ_1"), I_i_μ̃_Κ("i_2,μ̃_19584,Κ_1"), "i_1,μ̃_19584")("i_1,μ̃_19584");
  I_i_μ̃_Κ = TA::TSpArrayD();  // release
  I_i_i_Κ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19584"), C_ap1_μ̃("i_1,μ̃_19584;a_1"), "i_1;a_1")("i_1;a_1")) * (-1);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I3_i_μ̃("i_1,μ̃_19592") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_1,μ̃_19592;a_2"), t_ap1_i("i_1;a_2"), "i_1,μ̃_19592")("i_1,μ̃_19592");
  I_i_i_Κ("i_1,i_2,Κ_1") = TA::einsum(I3_i_μ̃("i_1,μ̃_19592"), g_i_μ̃_Κ("i_2,μ̃_19592,Κ_1"), "i_1,i_2,Κ_1")("i_1,i_2,Κ_1");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap2_Κ("i_2,i_3,Κ_1;a_3") = TA::einsum(g_i_μ̃_Κ("i_3,μ̃_19593,Κ_1"), C_μ̃_ap2("i_2,i_3,μ̃_19593;a_3"), "i_2,i_3,Κ_1;a_3")("i_2,i_3,Κ_1;a_3");
  I_i_i_i_ap2("i_1,i_2,i_3;a_3") = TA::einsum(I_i_i_Κ("i_1,i_2,Κ_1"), I_i_ap2_Κ("i_2,i_3,Κ_1;a_3"), "i_1,i_2,i_3;a_3")("i_1,i_2,i_3;a_3");
  I_i_ap2_Κ = ArrayToT();  // release
  I_i_i_Κ = TA::TSpArrayD();  // release
  I2_i_ap2("i_2,i_3,i_1;a_4") = TA::einsum(I_i_i_i_ap2("i_1,i_2,i_3;a_3"), t_ap2_ap2_i_i("i_3,i_2;a_3,a_4"), "i_2,i_3,i_1;a_4")("i_2,i_3,i_1;a_4");
  I_i_i_i_ap2 = ArrayToT();  // release
  I2_i_μ̃("i_1,μ̃_19595") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_2,i_3,i_1;a_4"), C_μ̃_ap2("i_2,i_3,μ̃_19595;a_4"), "i_1,μ̃_19595")("i_1,μ̃_19595");
  I2_i_ap2 = ArrayToT();  // release
  I_i_μ̃("i_1,μ̃_19594") = TA::einsum(I2_i_μ̃("i_1,μ̃_19595"), s_μ̃_μ̃("μ̃_19594,μ̃_19595"), "i_1,μ̃_19594")("i_1,μ̃_19594");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19594"), C_ap1_μ̃("i_1,μ̃_19594;a_1"), "i_1;a_1")("i_1;a_1")) * (-2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap2_Κ("i_1,i_2,Κ_1;a_2") = TA::einsum(g_i_μ̃_Κ("i_2,μ̃_19614,Κ_1"), C_μ̃_ap2("i_1,i_2,μ̃_19614;a_2"), "i_1,i_2,Κ_1;a_2")("i_1,i_2,Κ_1;a_2");
  I_i_i_ap2_μ̃("i_1,i_2,μ̃_19616;a_2") = TA::einsum(C_μ̃_ap2("i_1,i_2,μ̃_19616;a_3"), t_ap2_ap2_i_i("i_2,i_1;a_2,a_3"), "i_1,i_2,μ̃_19616;a_2")("i_1,i_2,μ̃_19616;a_2");
  I_i_μ̃_Κ("i_1,μ̃_19616,Κ_1") = TA::einsum<TA::DeNest::True>(I_i_ap2_Κ("i_1,i_2,Κ_1;a_2"), I_i_i_ap2_μ̃("i_1,i_2,μ̃_19616;a_2"), "i_1,μ̃_19616,Κ_1")("i_1,μ̃_19616,Κ_1");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_ap2_Κ = ArrayToT();  // release
  I_i_μ̃("i_1,μ̃_19615") = TA::einsum(I_i_μ̃_Κ("i_1,μ̃_19616,Κ_1"), g_μ̃_μ̃_Κ("μ̃_19615,μ̃_19616,Κ_1"), "i_1,μ̃_19615")("i_1,μ̃_19615");
  I_i_μ̃_Κ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19615"), C_ap1_μ̃("i_1,μ̃_19615;a_1"), "i_1;a_1")("i_1;a_1")) * (2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap2_Κ("i_1,i_2,Κ_1;a_2") = TA::einsum(g_i_μ̃_Κ("i_2,μ̃_19646,Κ_1"), C_μ̃_ap2("i_1,i_2,μ̃_19646;a_2"), "i_1,i_2,Κ_1;a_2")("i_1,i_2,Κ_1;a_2");
  I_i_i_ap2_μ̃("i_1,i_2,μ̃_19648;a_2") = TA::einsum(C_μ̃_ap2("i_1,i_2,μ̃_19648;a_3"), t_ap2_ap2_i_i("i_1,i_2;a_2,a_3"), "i_1,i_2,μ̃_19648;a_2")("i_1,i_2,μ̃_19648;a_2");
  I_i_μ̃_Κ("i_1,μ̃_19648,Κ_1") = TA::einsum<TA::DeNest::True>(I_i_ap2_Κ("i_1,i_2,Κ_1;a_2"), I_i_i_ap2_μ̃("i_1,i_2,μ̃_19648;a_2"), "i_1,μ̃_19648,Κ_1")("i_1,μ̃_19648,Κ_1");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_ap2_Κ = ArrayToT();  // release
  I_i_μ̃("i_1,μ̃_19647") = TA::einsum(I_i_μ̃_Κ("i_1,μ̃_19648,Κ_1"), g_μ̃_μ̃_Κ("μ̃_19647,μ̃_19648,Κ_1"), "i_1,μ̃_19647")("i_1,μ̃_19647");
  I_i_μ̃_Κ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19647"), C_ap1_μ̃("i_1,μ̃_19647;a_1"), "i_1;a_1")("i_1;a_1")) * (-1);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I3_i_μ̃("i_1,μ̃_19638") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_1,μ̃_19638;a_2"), t_ap1_i("i_1;a_2"), "i_1,μ̃_19638")("i_1,μ̃_19638");
  I_i_i_Κ("i_1,i_2,Κ_1") = TA::einsum(I3_i_μ̃("i_1,μ̃_19638"), g_i_μ̃_Κ("i_2,μ̃_19638,Κ_1"), "i_1,i_2,Κ_1")("i_1,i_2,Κ_1");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap2_Κ("i_2,i_3,Κ_1;a_3") = TA::einsum(g_i_μ̃_Κ("i_3,μ̃_19639,Κ_1"), C_μ̃_ap2("i_2,i_3,μ̃_19639;a_3"), "i_2,i_3,Κ_1;a_3")("i_2,i_3,Κ_1;a_3");
  I_i_i_i_ap2("i_1,i_2,i_3;a_3") = TA::einsum(I_i_i_Κ("i_1,i_2,Κ_1"), I_i_ap2_Κ("i_2,i_3,Κ_1;a_3"), "i_1,i_2,i_3;a_3")("i_1,i_2,i_3;a_3");
  I_i_ap2_Κ = ArrayToT();  // release
  I_i_i_Κ = TA::TSpArrayD();  // release
  I2_i_ap2("i_2,i_3,i_1;a_4") = TA::einsum(I_i_i_i_ap2("i_1,i_2,i_3;a_3"), t_ap2_ap2_i_i("i_2,i_3;a_3,a_4"), "i_2,i_3,i_1;a_4")("i_2,i_3,i_1;a_4");
  I_i_i_i_ap2 = ArrayToT();  // release
  I2_i_μ̃("i_1,μ̃_19641") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_2,i_3,i_1;a_4"), C_μ̃_ap2("i_2,i_3,μ̃_19641;a_4"), "i_1,μ̃_19641")("i_1,μ̃_19641");
  I2_i_ap2 = ArrayToT();  // release
  I_i_μ̃("i_1,μ̃_19640") = TA::einsum(I2_i_μ̃("i_1,μ̃_19641"), s_μ̃_μ̃("μ̃_19640,μ̃_19641"), "i_1,μ̃_19640")("i_1,μ̃_19640");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += TA::einsum(I_i_μ̃("i_1,μ̃_19640"), C_ap1_μ̃("i_1,μ̃_19640;a_1"), "i_1;a_1")("i_1;a_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I3_i_μ̃("i_2,μ̃_19620") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19620;a_3"), t_ap1_i("i_2;a_3"), "i_2,μ̃_19620")("i_2,μ̃_19620");
  I_i_i_Κ("i_2,i_3,Κ_1") = TA::einsum(I3_i_μ̃("i_2,μ̃_19620"), g_i_μ̃_Κ("i_3,μ̃_19620,Κ_1"), "i_2,i_3,Κ_1")("i_2,i_3,Κ_1");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_μ̃("i_3,μ̃_19619") = TA::einsum(I_i_i_Κ("i_2,i_3,Κ_1"), g_i_μ̃_Κ("i_2,μ̃_19619,Κ_1"), "i_3,μ̃_19619")("i_3,μ̃_19619");
  I_i_i_Κ = TA::TSpArrayD();  // release
  I3_i_μ̃("i_1,μ̃_19619") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_1,μ̃_19619;a_2"), t_ap1_i("i_1;a_2"), "i_1,μ̃_19619")("i_1,μ̃_19619");
  I_i_i("i_1,i_3") = TA::einsum(I2_i_μ̃("i_3,μ̃_19619"), I3_i_μ̃("i_1,μ̃_19619"), "i_1,i_3")("i_1,i_3");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I3_i_μ̃("i_3,μ̃_19622") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_3,μ̃_19622;a_4"), t_ap1_i("i_3;a_4"), "i_3,μ̃_19622")("i_3,μ̃_19622");
  I2_i_μ̃("i_3,μ̃_19621") = TA::einsum(I3_i_μ̃("i_3,μ̃_19622"), s_μ̃_μ̃("μ̃_19621,μ̃_19622"), "i_3,μ̃_19621")("i_3,μ̃_19621");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_1,μ̃_19621") = TA::einsum(I_i_i("i_1,i_3"), I2_i_μ̃("i_3,μ̃_19621"), "i_1,μ̃_19621")("i_1,μ̃_19621");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += TA::einsum(I_i_μ̃("i_1,μ̃_19621"), C_ap1_μ̃("i_1,μ̃_19621;a_1"), "i_1;a_1")("i_1;a_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I4_i_μ̃("i_3,μ̃_19634") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_3,μ̃_19634;a_2"), t_ap1_i("i_3;a_2"), "i_3,μ̃_19634")("i_3,μ̃_19634");
  I_i_i_Κ("i_2,i_3,Κ_1") = TA::einsum(I4_i_μ̃("i_3,μ̃_19634"), g_i_μ̃_Κ("i_2,μ̃_19634,Κ_1"), "i_2,i_3,Κ_1")("i_2,i_3,Κ_1");
  I4_i_μ̃ = TA::TSpArrayD();  // release
  I3_i_μ̃("i_2,μ̃_19635") = TA::einsum(I_i_i_Κ("i_2,i_3,Κ_1"), g_i_μ̃_Κ("i_3,μ̃_19635,Κ_1"), "i_2,μ̃_19635")("i_2,μ̃_19635");
  I_i_i_Κ = TA::TSpArrayD();  // release
  I3_i_ap2("i_1,i_2;a_3") = TA::einsum(I3_i_μ̃("i_2,μ̃_19635"), C_μ̃_ap2("i_1,i_2,μ̃_19635;a_3"), "i_1,i_2;a_3")("i_1,i_2;a_3");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_ap2("i_2,i_1;a_4") = TA::einsum(I3_i_ap2("i_1,i_2;a_3"), t_ap2_ap2_i_i("i_2,i_1;a_3,a_4"), "i_2,i_1;a_4")("i_2,i_1;a_4");
  I3_i_ap2 = ArrayToT();  // release
  I2_i_μ̃("i_1,μ̃_19637") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_2,i_1;a_4"), C_μ̃_ap2("i_1,i_2,μ̃_19637;a_4"), "i_1,μ̃_19637")("i_1,μ̃_19637");
  I2_i_ap2 = ArrayToT();  // release
  I_i_μ̃("i_1,μ̃_19636") = TA::einsum(I2_i_μ̃("i_1,μ̃_19637"), s_μ̃_μ̃("μ̃_19636,μ̃_19637"), "i_1,μ̃_19636")("i_1,μ̃_19636");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19636"), C_ap1_μ̃("i_1,μ̃_19636;a_1"), "i_1;a_1")("i_1;a_1")) * (-2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I4_i_μ̃("i_3,μ̃_19657") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_3,μ̃_19657;a_2"), t_ap1_i("i_3;a_2"), "i_3,μ̃_19657")("i_3,μ̃_19657");
  I_i_i_Κ("i_2,i_3,Κ_1") = TA::einsum(I4_i_μ̃("i_3,μ̃_19657"), g_i_μ̃_Κ("i_2,μ̃_19657,Κ_1"), "i_2,i_3,Κ_1")("i_2,i_3,Κ_1");
  I4_i_μ̃ = TA::TSpArrayD();  // release
  I3_i_μ̃("i_2,μ̃_19658") = TA::einsum(I_i_i_Κ("i_2,i_3,Κ_1"), g_i_μ̃_Κ("i_3,μ̃_19658,Κ_1"), "i_2,μ̃_19658")("i_2,μ̃_19658");
  I_i_i_Κ = TA::TSpArrayD();  // release
  I3_i_ap2("i_1,i_2;a_3") = TA::einsum(I3_i_μ̃("i_2,μ̃_19658"), C_μ̃_ap2("i_1,i_2,μ̃_19658;a_3"), "i_1,i_2;a_3")("i_1,i_2;a_3");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_ap2("i_2,i_1;a_4") = TA::einsum(I3_i_ap2("i_1,i_2;a_3"), t_ap2_ap2_i_i("i_1,i_2;a_3,a_4"), "i_2,i_1;a_4")("i_2,i_1;a_4");
  I3_i_ap2 = ArrayToT();  // release
  I2_i_μ̃("i_1,μ̃_19660") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_2,i_1;a_4"), C_μ̃_ap2("i_1,i_2,μ̃_19660;a_4"), "i_1,μ̃_19660")("i_1,μ̃_19660");
  I2_i_ap2 = ArrayToT();  // release
  I_i_μ̃("i_1,μ̃_19659") = TA::einsum(I2_i_μ̃("i_1,μ̃_19660"), s_μ̃_μ̃("μ̃_19659,μ̃_19660"), "i_1,μ̃_19659")("i_1,μ̃_19659");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += TA::einsum(I_i_μ̃("i_1,μ̃_19659"), C_ap1_μ̃("i_1,μ̃_19659;a_1"), "i_1;a_1")("i_1;a_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_μ̃_μ̃("i_2,i_3,μ̃_19596,μ̃_19597") = TA::einsum(g_i_μ̃_Κ("i_2,μ̃_19596,Κ_1"), g_i_μ̃_Κ("i_3,μ̃_19597,Κ_1"), "i_2,i_3,μ̃_19596,μ̃_19597")("i_2,i_3,μ̃_19596,μ̃_19597");
  I_i_i_ap2_μ̃("i_1,i_2,i_3,μ̃_19597;a_2") = TA::einsum(I_i_i_μ̃_μ̃("i_2,i_3,μ̃_19596,μ̃_19597"), C_μ̃_ap2("i_1,i_2,μ̃_19596;a_2"), "i_1,i_2,i_3,μ̃_19597;a_2")("i_1,i_2,i_3,μ̃_19597;a_2");
  I_i_i_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2("i_1,i_2,i_3;a_2,a_3") = TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,i_3,μ̃_19597;a_2"), C_μ̃_ap2("i_1,i_2,μ̃_19597;a_3"), "i_1,i_2,i_3;a_2,a_3")("i_1,i_2,i_3;a_2,a_3");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_i("i_1,i_3") = TA::einsum<TA::DeNest::True>(I_i_i_ap2_ap2("i_1,i_2,i_3;a_2,a_3"), t_ap2_ap2_i_i("i_1,i_2;a_2,a_3"), "i_1,i_3")("i_1,i_3");
  I_i_i_ap2_ap2 = ArrayToT();  // release
  I3_i_μ̃("i_3,μ̃_19599") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_3,μ̃_19599;a_4"), t_ap1_i("i_3;a_4"), "i_3,μ̃_19599")("i_3,μ̃_19599");
  I2_i_μ̃("i_3,μ̃_19598") = TA::einsum(I3_i_μ̃("i_3,μ̃_19599"), s_μ̃_μ̃("μ̃_19598,μ̃_19599"), "i_3,μ̃_19598")("i_3,μ̃_19598");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_1,μ̃_19598") = TA::einsum(I_i_i("i_1,i_3"), I2_i_μ̃("i_3,μ̃_19598"), "i_1,μ̃_19598")("i_1,μ̃_19598");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += TA::einsum(I_i_μ̃("i_1,μ̃_19598"), C_ap1_μ̃("i_1,μ̃_19598;a_1"), "i_1;a_1")("i_1;a_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_μ̃_μ̃("i_2,i_3,μ̃_19649,μ̃_19650") = TA::einsum(g_i_μ̃_Κ("i_2,μ̃_19649,Κ_1"), g_i_μ̃_Κ("i_3,μ̃_19650,Κ_1"), "i_2,i_3,μ̃_19649,μ̃_19650")("i_2,i_3,μ̃_19649,μ̃_19650");
  I_i_i_ap2_μ̃("i_1,i_2,i_3,μ̃_19650;a_2") = TA::einsum(I_i_i_μ̃_μ̃("i_2,i_3,μ̃_19649,μ̃_19650"), C_μ̃_ap2("i_1,i_3,μ̃_19649;a_2"), "i_1,i_2,i_3,μ̃_19650;a_2")("i_1,i_2,i_3,μ̃_19650;a_2");
  I_i_i_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2("i_1,i_2,i_3;a_2,a_3") = TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,i_3,μ̃_19650;a_2"), C_μ̃_ap2("i_1,i_3,μ̃_19650;a_3"), "i_1,i_2,i_3;a_2,a_3")("i_1,i_2,i_3;a_2,a_3");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_i("i_1,i_2") = TA::einsum<TA::DeNest::True>(I_i_i_ap2_ap2("i_1,i_2,i_3;a_2,a_3"), t_ap2_ap2_i_i("i_1,i_3;a_2,a_3"), "i_1,i_2")("i_1,i_2");
  I_i_i_ap2_ap2 = ArrayToT();  // release
  I3_i_μ̃("i_2,μ̃_19652") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19652;a_4"), t_ap1_i("i_2;a_4"), "i_2,μ̃_19652")("i_2,μ̃_19652");
  I2_i_μ̃("i_2,μ̃_19651") = TA::einsum(I3_i_μ̃("i_2,μ̃_19652"), s_μ̃_μ̃("μ̃_19651,μ̃_19652"), "i_2,μ̃_19651")("i_2,μ̃_19651");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_1,μ̃_19651") = TA::einsum(I_i_i("i_1,i_2"), I2_i_μ̃("i_2,μ̃_19651"), "i_1,μ̃_19651")("i_1,μ̃_19651");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19651"), C_ap1_μ̃("i_1,μ̃_19651;a_1"), "i_1;a_1")("i_1;a_1")) * (-2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_i_μ̃("i_1,i_2,i_3,μ̃_19608") = TA::einsum(g_i_i_Κ("i_2,i_1,Κ_1"), g_i_μ̃_Κ("i_3,μ̃_19608,Κ_1"), "i_1,i_2,i_3,μ̃_19608")("i_1,i_2,i_3,μ̃_19608");
  I2_i_μ̃("i_2,μ̃_19608") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19608;a_2"), t_ap1_i("i_2;a_2"), "i_2,μ̃_19608")("i_2,μ̃_19608");
  I_i_i("i_1,i_3") = TA::einsum(I_i_i_i_μ̃("i_1,i_2,i_3,μ̃_19608"), I2_i_μ̃("i_2,μ̃_19608"), "i_1,i_3")("i_1,i_3");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_i_μ̃ = TA::TSpArrayD();  // release
  I3_i_μ̃("i_3,μ̃_19610") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_3,μ̃_19610;a_3"), t_ap1_i("i_3;a_3"), "i_3,μ̃_19610")("i_3,μ̃_19610");
  I2_i_μ̃("i_3,μ̃_19609") = TA::einsum(I3_i_μ̃("i_3,μ̃_19610"), s_μ̃_μ̃("μ̃_19609,μ̃_19610"), "i_3,μ̃_19609")("i_3,μ̃_19609");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_1,μ̃_19609") = TA::einsum(I_i_i("i_1,i_3"), I2_i_μ̃("i_3,μ̃_19609"), "i_1,μ̃_19609")("i_1,μ̃_19609");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += TA::einsum(I_i_μ̃("i_1,μ̃_19609"), C_ap1_μ̃("i_1,μ̃_19609;a_1"), "i_1;a_1")("i_1;a_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_ap1("i_1,i_2;a_2") = TA::einsum(f_i_μ̃("i_2,μ̃_19630"), C_μ̃_ap1("i_1,μ̃_19630;a_2"), "i_1,i_2;a_2")("i_1,i_2;a_2");
  I_i_i("i_1,i_2") = TA::einsum<TA::DeNest::True>(I2_i_ap1("i_1,i_2;a_2"), t_ap1_i("i_1;a_2"), "i_1,i_2")("i_1,i_2");
  I2_i_ap1 = ArrayToT();  // release
  I3_i_μ̃("i_2,μ̃_19632") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19632;a_3"), t_ap1_i("i_2;a_3"), "i_2,μ̃_19632")("i_2,μ̃_19632");
  I2_i_μ̃("i_2,μ̃_19631") = TA::einsum(I3_i_μ̃("i_2,μ̃_19632"), s_μ̃_μ̃("μ̃_19631,μ̃_19632"), "i_2,μ̃_19631")("i_2,μ̃_19631");
  I3_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_1,μ̃_19631") = TA::einsum(I_i_i("i_1,i_2"), I2_i_μ̃("i_2,μ̃_19631"), "i_1,μ̃_19631")("i_1,μ̃_19631");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19631"), C_ap1_μ̃("i_1,μ̃_19631;a_1"), "i_1;a_1")("i_1;a_1")) * (-1);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_μ̃_μ̃("i_1,i_2,μ̃_19606,μ̃_19607") = TA::einsum(g_i_i_Κ("i_2,i_1,Κ_1"), g_μ̃_μ̃_Κ("μ̃_19606,μ̃_19607,Κ_1"), "i_1,i_2,μ̃_19606,μ̃_19607")("i_1,i_2,μ̃_19606,μ̃_19607");
  I2_i_μ̃("i_2,μ̃_19607") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19607;a_2"), t_ap1_i("i_2;a_2"), "i_2,μ̃_19607")("i_2,μ̃_19607");
  I_i_μ̃("i_1,μ̃_19606") = TA::einsum(I_i_i_μ̃_μ̃("i_1,i_2,μ̃_19606,μ̃_19607"), I2_i_μ̃("i_2,μ̃_19607"), "i_1,μ̃_19606")("i_1,μ̃_19606");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19606"), C_ap1_μ̃("i_1,μ̃_19606;a_1"), "i_1;a_1")("i_1;a_1")) * (-1);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_μ̃("i_1,μ̃_19618") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_1,μ̃_19618;a_2"), t_ap1_i("i_1;a_2"), "i_1,μ̃_19618")("i_1,μ̃_19618");
  I_i_μ̃("i_1,μ̃_19617") = TA::einsum(I2_i_μ̃("i_1,μ̃_19618"), f_μ̃_μ̃("μ̃_19617,μ̃_19618"), "i_1,μ̃_19617")("i_1,μ̃_19617");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += TA::einsum(I_i_μ̃("i_1,μ̃_19617"), C_ap1_μ̃("i_1,μ̃_19617;a_1"), "i_1;a_1")("i_1;a_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_i_μ̃("i_1,i_2,i_3,μ̃_19589") = TA::einsum(g_i_i_Κ("i_2,i_1,Κ_1"), g_i_μ̃_Κ("i_3,μ̃_19589,Κ_1"), "i_1,i_2,i_3,μ̃_19589")("i_1,i_2,i_3,μ̃_19589");
  I_i_i_i_ap2("i_1,i_2,i_3;a_2") = TA::einsum(I_i_i_i_μ̃("i_1,i_2,i_3,μ̃_19589"), C_μ̃_ap2("i_2,i_3,μ̃_19589;a_2"), "i_1,i_2,i_3;a_2")("i_1,i_2,i_3;a_2");
  I_i_i_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_ap2("i_2,i_3,i_1;a_3") = TA::einsum(I_i_i_i_ap2("i_1,i_2,i_3;a_2"), t_ap2_ap2_i_i("i_3,i_2;a_2,a_3"), "i_2,i_3,i_1;a_3")("i_2,i_3,i_1;a_3");
  I_i_i_i_ap2 = ArrayToT();  // release
  I2_i_μ̃("i_1,μ̃_19591") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_2,i_3,i_1;a_3"), C_μ̃_ap2("i_2,i_3,μ̃_19591;a_3"), "i_1,μ̃_19591")("i_1,μ̃_19591");
  I2_i_ap2 = ArrayToT();  // release
  I_i_μ̃("i_1,μ̃_19590") = TA::einsum(I2_i_μ̃("i_1,μ̃_19591"), s_μ̃_μ̃("μ̃_19590,μ̃_19591"), "i_1,μ̃_19590")("i_1,μ̃_19590");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19590"), C_ap1_μ̃("i_1,μ̃_19590;a_1"), "i_1;a_1")("i_1;a_1")) * (-2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_i_μ̃("i_1,i_2,i_3,μ̃_19611") = TA::einsum(g_i_i_Κ("i_2,i_1,Κ_1"), g_i_μ̃_Κ("i_3,μ̃_19611,Κ_1"), "i_1,i_2,i_3,μ̃_19611")("i_1,i_2,i_3,μ̃_19611");
  I_i_i_i_ap2("i_1,i_2,i_3;a_2") = TA::einsum(I_i_i_i_μ̃("i_1,i_2,i_3,μ̃_19611"), C_μ̃_ap2("i_2,i_3,μ̃_19611;a_2"), "i_1,i_2,i_3;a_2")("i_1,i_2,i_3;a_2");
  I_i_i_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_ap2("i_2,i_3,i_1;a_3") = TA::einsum(I_i_i_i_ap2("i_1,i_2,i_3;a_2"), t_ap2_ap2_i_i("i_2,i_3;a_2,a_3"), "i_2,i_3,i_1;a_3")("i_2,i_3,i_1;a_3");
  I_i_i_i_ap2 = ArrayToT();  // release
  I2_i_μ̃("i_1,μ̃_19613") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_2,i_3,i_1;a_3"), C_μ̃_ap2("i_2,i_3,μ̃_19613;a_3"), "i_1,μ̃_19613")("i_1,μ̃_19613");
  I2_i_ap2 = ArrayToT();  // release
  I_i_μ̃("i_1,μ̃_19612") = TA::einsum(I2_i_μ̃("i_1,μ̃_19613"), s_μ̃_μ̃("μ̃_19612,μ̃_19613"), "i_1,μ̃_19612")("i_1,μ̃_19612");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += TA::einsum(I_i_μ̃("i_1,μ̃_19612"), C_ap1_μ̃("i_1,μ̃_19612;a_1"), "i_1;a_1")("i_1;a_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I3_i_ap2("i_1,i_2;a_2") = TA::einsum(f_i_μ̃("i_2,μ̃_19600"), C_μ̃_ap2("i_1,i_2,μ̃_19600;a_2"), "i_1,i_2;a_2")("i_1,i_2;a_2");
  I2_i_ap2("i_2,i_1;a_3") = TA::einsum(I3_i_ap2("i_1,i_2;a_2"), t_ap2_ap2_i_i("i_1,i_2;a_2,a_3"), "i_2,i_1;a_3")("i_2,i_1;a_3");
  I3_i_ap2 = ArrayToT();  // release
  I2_i_μ̃("i_1,μ̃_19602") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_2,i_1;a_3"), C_μ̃_ap2("i_1,i_2,μ̃_19602;a_3"), "i_1,μ̃_19602")("i_1,μ̃_19602");
  I2_i_ap2 = ArrayToT();  // release
  I_i_μ̃("i_1,μ̃_19601") = TA::einsum(I2_i_μ̃("i_1,μ̃_19602"), s_μ̃_μ̃("μ̃_19601,μ̃_19602"), "i_1,μ̃_19601")("i_1,μ̃_19601");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19601"), C_ap1_μ̃("i_1,μ̃_19601;a_1"), "i_1;a_1")("i_1;a_1")) * (-1);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I3_i_ap2("i_1,i_2;a_2") = TA::einsum(f_i_μ̃("i_2,μ̃_19603"), C_μ̃_ap2("i_1,i_2,μ̃_19603;a_2"), "i_1,i_2;a_2")("i_1,i_2;a_2");
  I2_i_ap2("i_2,i_1;a_3") = TA::einsum(I3_i_ap2("i_1,i_2;a_2"), t_ap2_ap2_i_i("i_2,i_1;a_2,a_3"), "i_2,i_1;a_3")("i_2,i_1;a_3");
  I3_i_ap2 = ArrayToT();  // release
  I2_i_μ̃("i_1,μ̃_19605") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_2,i_1;a_3"), C_μ̃_ap2("i_1,i_2,μ̃_19605;a_3"), "i_1,μ̃_19605")("i_1,μ̃_19605");
  I2_i_ap2 = ArrayToT();  // release
  I_i_μ̃("i_1,μ̃_19604") = TA::einsum(I2_i_μ̃("i_1,μ̃_19605"), s_μ̃_μ̃("μ̃_19604,μ̃_19605"), "i_1,μ̃_19604")("i_1,μ̃_19604");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19604"), C_ap1_μ̃("i_1,μ̃_19604;a_1"), "i_1;a_1")("i_1;a_1")) * (2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += TA::einsum(f_μ̃_i("μ̃_19633,i_1"), C_ap1_μ̃("i_1,μ̃_19633;a_1"), "i_1;a_1")("i_1;a_1");
  return I_i_ap1;
}

