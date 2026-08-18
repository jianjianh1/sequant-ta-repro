// AUX-Κ-BATCHING VARIANT of generated_t2_residual.cpp (do NOT regenerate this
// file from the codegen — it is a hand-derived experiment copy). Byte-for-byte
// identical to the shipped generated T2 residual EXCEPT the DF half-transform
// block (lines 487-491 of the original) is gated: when the env var
// SPTC_AUX_TARGET_SIZE > 0, it is computed by accumulate_df_halftransform_batched()
// (aux_k_batching.h), which streams the aux index Κ in tile-aligned batches so
// the giant μ̃Κ intermediate is never fully formed. SPTC_AUX_TARGET_SIZE unset
// or 0 → the original einsum block runs verbatim (bit-identical to the shipped
// binary). See aux_k_batching.h.
//
// Original header:
// AUTO-GENERATED (2026-07-19) by SeQuant's native TiledArrayGenerator from the
// full cck.ipp-matching closed-shell CSV-CCSD T2 residual derivation (55 terms).
#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>
#include <cmath>
#include <cstdlib>
#include "ta_tensors.h"
#include "aux_k_batching.h"

ArrayToT whole_t2_residual(const ArrayToT& C_μ̃_ap1, const TA::TSpArrayD& g_i_μ̃_Κ, const ArrayToT& t_ap1_i, const ArrayToT& C_ap2_μ̃, const ArrayToT& C_μ̃_ap2, const TA::TSpArrayD& g_μ̃_μ̃_Κ, const ArrayToT& t_ap2_ap2_i_i, const TA::TSpArrayD& s_μ̃_μ̃, const TA::TSpArrayD& g_i_i_Κ, const TA::TSpArrayD& f_i_μ̃, const TA::TSpArrayD& g_μ̃_i_Κ, const TA::TSpArrayD& f_i_i, const TA::TSpArrayD& f_μ̃_μ̃) {
  TA::TSpArrayD CSE1_Κ;
  TA::TSpArrayD I_i_μ̃;
  ArrayToT I_ap2_ap2;
  ArrayToT I_ap2_μ̃;
  TA::TSpArrayD I_μ̃_μ̃;
  ArrayToT I_i_i_ap2_ap2;
  TA::TSpArrayD CSE2_i_μ̃;
  TA::TSpArrayD I_Κ;
  TA::TSpArrayD CSE3_i_μ̃;
  ArrayToT CSE4_i_i_i_ap2_ap2;
  ArrayToT CSE5_i_i_i_ap2_ap2;
  TA::TSpArrayD I_i_i;
  ArrayToT I2_i_i_ap2_ap2;
  ArrayToT CSE6_i_i_i_ap2;
  TA::TSpArrayD I2_i_μ̃;
  ArrayToT I_i_ap2;
  ArrayToT I_i_i_i_ap2;
  ArrayToT CSE7_i_i_i_ap2_ap2;
  ArrayToT I_i_ap1;
  ArrayToT CSE8_i_i_i_ap2_ap2;
  ArrayToT I_i_i_ap2_μ̃;
  TA::TSpArrayD I_i_i_μ̃_μ̃;
  ArrayToT CSE9_i_i_i_i_ap2_ap2;
  TA::TSpArrayD CSE10_i_i_μ̃_μ̃;
  TA::TSpArrayD I_i_i_Κ;
  ArrayToT CSE11_i_i_ap2_μ̃;
  ArrayToT I2_i_i_ap2_μ̃;
  TA::TSpArrayD CSE12_i_i_i_i;
  TA::TSpArrayD I2_i_i_Κ;
  TA::TSpArrayD CSE13_i_i_μ̃_μ̃;
  ArrayToT CSE14_i_i_ap2_μ̃;
  TA::TSpArrayD CSE15_i_i_i_i;
  TA::TSpArrayD CSE16_i_i_i_i;
  TA::TSpArrayD I2_i_i_μ̃_μ̃;
  TA::TSpArrayD I_i_i_i_μ̃;
  TA::TSpArrayD I_i_μ̃_μ̃_μ̃;
  TA::TSpArrayD CSE17_i_i_i_i;
  TA::TSpArrayD CSE18_i_μ̃_Κ;
  ArrayToT I_i_ap2_Κ;
  ArrayToT CSE19_i_i_ap2_Κ;
  TA::TSpArrayD I_i_μ̃_Κ;
  ArrayToT CSE20_i_i_ap2_Κ;
  TA::TSpArrayD CSE21_i_μ̃_Κ;
  ArrayToT CSE22_i_i_ap2_Κ;
  ArrayToT I2_ap2_μ̃;
  TA::TSpArrayD I2_μ̃_μ̃;
  ArrayToT CSE23_i_i_i_ap2_ap2;
  ArrayToT I2_i_ap2_Κ;
  ArrayToT CSE24_i_i_i_i_ap2;
  ArrayToT I2_i_i_i_ap2;
  ArrayToT CSE25_i_i_i_i_ap2;
  TA::TSpArrayD CSE26_i_i_Κ;
  ArrayToT CSE27_i_i_ap2_Κ;
  TA::TSpArrayD CSE28_i_μ̃_Κ;
  TA::TSpArrayD CSE29_i_μ̃;
  ArrayToT CSE30_i_i_i_i_ap2;
  ArrayToT CSE31_i_i_i_i_ap2;
  TA::TSpArrayD CSE32_i_i_μ̃_μ̃;
  TA::TSpArrayD CSE33_i_i_i_μ̃;
  ArrayToT CSE34_i_i_i_ap2_μ̃;
  ArrayToT I3_i_i_ap2_ap2;
  TA::TSpArrayD CSE35_i_i_μ̃_μ̃;
  ArrayToT CSE36_i_i_i_ap2_ap2;
  ArrayToT CSE37_i_i_ap2_ap2_Κ;
  ArrayToT I_ap2_μ̃_Κ;
  ArrayToT I_i_i_ap2_ap2_Κ;
  ArrayToT I_i_i_ap2_ap2_RANKFIX1;
  I_i_μ̃("i_3,μ̃_19661") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_3,μ̃_19661;a_3"), t_ap1_i("i_3;a_3"), "i_3,μ̃_19661")("i_3,μ̃_19661");
  CSE1_Κ("Κ_1") = TA::einsum(I_i_μ̃("i_3,μ̃_19661"), g_i_μ̃_Κ("i_3,μ̃_19661,Κ_1"), "Κ_1")("Κ_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_μ̃_μ̃("μ̃_19662,μ̃_19663") = TA::einsum(CSE1_Κ("Κ_1"), g_μ̃_μ̃_Κ("μ̃_19662,μ̃_19663,Κ_1"), "μ̃_19662,μ̃_19663")("μ̃_19662,μ̃_19663");
  I_ap2_μ̃("i_1,i_2,μ̃_19663;a_2") = TA::einsum(I_μ̃_μ̃("μ̃_19662,μ̃_19663"), C_ap2_μ̃("i_1,i_2,μ̃_19662;a_2"), "i_1,i_2,μ̃_19663;a_2")("i_1,i_2,μ̃_19663;a_2");
  I_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_ap2_ap2("i_1,i_2;a_2,a_4") = TA::einsum(I_ap2_μ̃("i_1,i_2,μ̃_19663;a_2"), C_μ̃_ap2("i_1,i_2,μ̃_19663;a_4"), "i_1,i_2;a_2,a_4")("i_1,i_2;a_2,a_4");
  I_ap2_μ̃ = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") = (TA::einsum(I_ap2_ap2("i_1,i_2;a_2,a_4"), t_ap2_ap2_i_i("i_1,i_2;a_1,a_4"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (4);
  I_ap2_ap2 = ArrayToT();  // release
  I_i_μ̃("i_3,μ̃_19712") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_3,μ̃_19712;a_3"), t_ap1_i("i_3;a_3"), "i_3,μ̃_19712")("i_3,μ̃_19712");
  I_Κ("Κ_1") = TA::einsum(I_i_μ̃("i_3,μ̃_19712"), g_i_μ̃_Κ("i_3,μ̃_19712,Κ_1"), "Κ_1")("Κ_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  CSE2_i_μ̃("i_4,μ̃_19713") = TA::einsum(I_Κ("Κ_1"), g_i_μ̃_Κ("i_4,μ̃_19713,Κ_1"), "i_4,μ̃_19713")("i_4,μ̃_19713");
  I_Κ = TA::TSpArrayD();  // release
  CSE3_i_μ̃("i_2,μ̃_19713") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19713;a_4"), t_ap1_i("i_2;a_4"), "i_2,μ̃_19713")("i_2,μ̃_19713");
  I_ap2_μ̃("i_1,i_2,μ̃_19717;a_2") = TA::einsum(s_μ̃_μ̃("μ̃_19716,μ̃_19717"), C_ap2_μ̃("i_1,i_2,μ̃_19716;a_2"), "i_1,i_2,μ̃_19717;a_2")("i_1,i_2,μ̃_19717;a_2");
  CSE4_i_i_i_ap2_ap2("i_4,i_2,i_1;a_2,a_6") = TA::einsum(I_ap2_μ̃("i_1,i_2,μ̃_19717;a_2"), C_μ̃_ap2("i_1,i_4,μ̃_19717;a_6"), "i_4,i_2,i_1;a_2,a_6")("i_4,i_2,i_1;a_2,a_6");
  I_ap2_μ̃ = ArrayToT();  // release
  I_ap2_μ̃("i_1,i_2,μ̃_19715;a_1") = TA::einsum(s_μ̃_μ̃("μ̃_19714,μ̃_19715"), C_ap2_μ̃("i_1,i_2,μ̃_19714;a_1"), "i_1,i_2,μ̃_19715;a_1")("i_1,i_2,μ̃_19715;a_1");
  I_ap2_ap2("i_1,i_2,i_4;a_1,a_5") = TA::einsum(I_ap2_μ̃("i_1,i_2,μ̃_19715;a_1"), C_μ̃_ap2("i_1,i_4,μ̃_19715;a_5"), "i_1,i_2,i_4;a_1,a_5")("i_1,i_2,i_4;a_1,a_5");
  I_ap2_μ̃ = ArrayToT();  // release
  CSE5_i_i_i_ap2_ap2("i_2,i_4,i_1;a_1,a_6") = TA::einsum(I_ap2_ap2("i_1,i_2,i_4;a_1,a_5"), t_ap2_ap2_i_i("i_1,i_4;a_5,a_6"), "i_2,i_4,i_1;a_1,a_6")("i_2,i_4,i_1;a_1,a_6");
  I_ap2_ap2 = ArrayToT();  // release
  I_i_i("i_2,i_4") = TA::einsum(CSE2_i_μ̃("i_4,μ̃_19713"), CSE3_i_μ̃("i_2,μ̃_19713"), "i_2,i_4")("i_2,i_4");
  I2_i_i_ap2_ap2("i_1,i_2,i_4;a_2,a_6") = TA::einsum(I_i_i("i_2,i_4"), CSE4_i_i_i_ap2_ap2("i_4,i_2,i_1;a_2,a_6"), "i_1,i_2,i_4;a_2,a_6")("i_1,i_2,i_4;a_2,a_6");
  I_i_i = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I2_i_i_ap2_ap2("i_1,i_2,i_4;a_2,a_6"), CSE5_i_i_i_ap2_ap2("i_2,i_4,i_1;a_1,a_6"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-4);
  I2_i_i_ap2_ap2 = ArrayToT();  // release
  I2_i_μ̃("i_4,μ̃_19870") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_4,μ̃_19870;a_5"), t_ap1_i("i_4;a_5"), "i_4,μ̃_19870")("i_4,μ̃_19870");
  I_i_μ̃("i_4,μ̃_19869") = TA::einsum(I2_i_μ̃("i_4,μ̃_19870"), s_μ̃_μ̃("μ̃_19869,μ̃_19870"), "i_4,μ̃_19869")("i_4,μ̃_19869");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  CSE6_i_i_i_ap2("i_2,i_1,i_4;a_2") = TA::einsum(I_i_μ̃("i_4,μ̃_19869"), C_ap2_μ̃("i_1,i_2,μ̃_19869;a_2"), "i_2,i_1,i_4;a_2")("i_2,i_1,i_4;a_2");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap2("i_1,i_2,i_4;a_4") = TA::einsum(CSE2_i_μ̃("i_4,μ̃_19868"), C_μ̃_ap2("i_1,i_2,μ̃_19868;a_4"), "i_1,i_2,i_4;a_4")("i_1,i_2,i_4;a_4");
  I_i_i_i_ap2("i_1,i_2,i_4;a_1") = TA::einsum(I_i_ap2("i_1,i_2,i_4;a_4"), t_ap2_ap2_i_i("i_1,i_2;a_1,a_4"), "i_1,i_2,i_4;a_1")("i_1,i_2,i_4;a_1");
  I_i_ap2 = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_i_ap2("i_1,i_2,i_4;a_1"), CSE6_i_i_i_ap2("i_2,i_1,i_4;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-4);
  I_i_i_i_ap2 = ArrayToT();  // release
  I_ap2_μ̃("i_1,i_2,μ̃_19759;a_1") = TA::einsum(s_μ̃_μ̃("μ̃_19758,μ̃_19759"), C_ap2_μ̃("i_1,i_2,μ̃_19758;a_1"), "i_1,i_2,μ̃_19759;a_1")("i_1,i_2,μ̃_19759;a_1");
  I_ap2_ap2("i_1,i_2,i_3;a_1,a_4") = TA::einsum(I_ap2_μ̃("i_1,i_2,μ̃_19759;a_1"), C_μ̃_ap2("i_2,i_3,μ̃_19759;a_4"), "i_1,i_2,i_3;a_1,a_4")("i_1,i_2,i_3;a_1,a_4");
  I_ap2_μ̃ = ArrayToT();  // release
  CSE7_i_i_i_ap2_ap2("i_1,i_3,i_2;a_1,a_5") = TA::einsum(I_ap2_ap2("i_1,i_2,i_3;a_1,a_4"), t_ap2_ap2_i_i("i_3,i_2;a_4,a_5"), "i_1,i_3,i_2;a_1,a_5")("i_1,i_3,i_2;a_1,a_5");
  I_ap2_ap2 = ArrayToT();  // release
  I_i_i("i_1,i_3") = TA::einsum(g_i_i_Κ("i_3,i_1,Κ_1"), CSE1_Κ("Κ_1"), "i_1,i_3")("i_1,i_3");
  I2_i_i_ap2_ap2("i_2,i_1,i_3;a_2,a_5") = TA::einsum(I_i_i("i_1,i_3"), CSE4_i_i_i_ap2_ap2("i_3,i_1,i_2;a_2,a_5"), "i_2,i_1,i_3;a_2,a_5")("i_2,i_1,i_3;a_2,a_5");
  I_i_i = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I2_i_i_ap2_ap2("i_2,i_1,i_3;a_2,a_5"), CSE7_i_i_i_ap2_ap2("i_1,i_3,i_2;a_1,a_5"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-4);
  I2_i_i_ap2_ap2 = ArrayToT();  // release
  I_i_ap1("i_1,i_3;a_3") = TA::einsum(f_i_μ̃("i_3,μ̃_19787"), C_μ̃_ap1("i_1,μ̃_19787;a_3"), "i_1,i_3;a_3")("i_1,i_3;a_3");
  I_i_i("i_1,i_3") = TA::einsum<TA::DeNest::True>(I_i_ap1("i_1,i_3;a_3"), t_ap1_i("i_1;a_3"), "i_1,i_3")("i_1,i_3");
  I_i_ap1 = ArrayToT();  // release
  I2_i_i_ap2_ap2("i_2,i_1,i_3;a_2,a_5") = TA::einsum(I_i_i("i_1,i_3"), CSE4_i_i_i_ap2_ap2("i_3,i_1,i_2;a_2,a_5"), "i_2,i_1,i_3;a_2,a_5")("i_2,i_1,i_3;a_2,a_5");
  I_i_i = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I2_i_i_ap2_ap2("i_2,i_1,i_3;a_2,a_5"), CSE7_i_i_i_ap2_ap2("i_1,i_3,i_2;a_1,a_5"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-2);
  I2_i_i_ap2_ap2 = ArrayToT();  // release
  I_i_i_μ̃_μ̃("i_3,i_4,μ̃_19851,μ̃_19852") = TA::einsum(g_i_μ̃_Κ("i_3,μ̃_19851,Κ_1"), g_i_μ̃_Κ("i_4,μ̃_19852,Κ_1"), "i_3,i_4,μ̃_19851,μ̃_19852")("i_3,i_4,μ̃_19851,μ̃_19852");
  I_i_i_ap2_μ̃("i_1,i_3,i_4,μ̃_19852;a_3") = TA::einsum(I_i_i_μ̃_μ̃("i_3,i_4,μ̃_19851,μ̃_19852"), C_μ̃_ap2("i_1,i_4,μ̃_19851;a_3"), "i_1,i_3,i_4,μ̃_19852;a_3")("i_1,i_3,i_4,μ̃_19852;a_3");
  I_i_i_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2_RANKFIX1("i_1,i_2,i_3,i_4;a_3,a_4") = TA::einsum(I_i_i_ap2_μ̃("i_1,i_3,i_4,μ̃_19852;a_3"), C_μ̃_ap2("i_2,i_3,μ̃_19852;a_4"), "i_1,i_2,i_3,i_4;a_3,a_4")("i_1,i_2,i_3,i_4;a_3,a_4");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_ap2_μ̃("i_1,i_2,μ̃_19854;a_1") = TA::einsum(s_μ̃_μ̃("μ̃_19853,μ̃_19854"), C_ap2_μ̃("i_1,i_2,μ̃_19853;a_1"), "i_1,i_2,μ̃_19854;a_1")("i_1,i_2,μ̃_19854;a_1");
  I_ap2_ap2("i_1,i_2,i_4;a_1,a_5") = TA::einsum(I_ap2_μ̃("i_1,i_2,μ̃_19854;a_1"), C_μ̃_ap2("i_1,i_4,μ̃_19854;a_5"), "i_1,i_2,i_4;a_1,a_5")("i_1,i_2,i_4;a_1,a_5");
  I_ap2_μ̃ = ArrayToT();  // release
  I2_i_i_ap2_ap2("i_2,i_1,i_4;a_1,a_3") = TA::einsum(I_ap2_ap2("i_1,i_2,i_4;a_1,a_5"), t_ap2_ap2_i_i("i_4,i_1;a_3,a_5"), "i_2,i_1,i_4;a_1,a_3")("i_2,i_1,i_4;a_1,a_3");
  I_ap2_ap2 = ArrayToT();  // release
  CSE8_i_i_i_ap2_ap2("i_2,i_3,i_1;a_1,a_4") = TA::einsum(I_i_i_ap2_ap2_RANKFIX1("i_1,i_2,i_3,i_4;a_3,a_4"), I2_i_i_ap2_ap2("i_2,i_1,i_4;a_1,a_3"), "i_2,i_3,i_1;a_1,a_4")("i_2,i_3,i_1;a_1,a_4");
  I2_i_i_ap2_ap2 = ArrayToT();  // release
  I_i_i_ap2_ap2_RANKFIX1 = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") = (TA::einsum(CSE8_i_i_i_ap2_ap2("i_2,i_3,i_1;a_1,a_4"), CSE7_i_i_i_ap2_ap2("i_1,i_3,i_2;a_2,a_4"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (2);
  I_i_i_μ̃_μ̃("i_3,i_4,μ̃_19861,μ̃_19862") = TA::einsum(g_i_μ̃_Κ("i_3,μ̃_19861,Κ_1"), g_i_μ̃_Κ("i_4,μ̃_19862,Κ_1"), "i_3,i_4,μ̃_19861,μ̃_19862")("i_3,i_4,μ̃_19861,μ̃_19862");
  I_i_i_ap2_μ̃("i_2,i_3,i_4,μ̃_19862;a_3") = TA::einsum(I_i_i_μ̃_μ̃("i_3,i_4,μ̃_19861,μ̃_19862"), C_μ̃_ap2("i_2,i_4,μ̃_19861;a_3"), "i_2,i_3,i_4,μ̃_19862;a_3")("i_2,i_3,i_4,μ̃_19862;a_3");
  I_i_i_μ̃_μ̃ = TA::TSpArrayD();  // release
  CSE9_i_i_i_i_ap2_ap2("i_1,i_2,i_3,i_4;a_4,a_3") = TA::einsum(I_i_i_ap2_μ̃("i_2,i_3,i_4,μ̃_19862;a_3"), C_μ̃_ap2("i_1,i_3,μ̃_19862;a_4"), "i_1,i_2,i_3,i_4;a_4,a_3")("i_1,i_2,i_3,i_4;a_4,a_3");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I2_i_i_ap2_ap2("i_1,i_2,i_3;a_1,a_4") = TA::einsum(CSE9_i_i_i_i_ap2_ap2("i_1,i_2,i_3,i_4;a_4,a_3"), CSE7_i_i_i_ap2_ap2("i_1,i_4,i_2;a_1,a_3"), "i_1,i_2,i_3;a_1,a_4")("i_1,i_2,i_3;a_1,a_4");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += TA::einsum(I2_i_i_ap2_ap2("i_1,i_2,i_3;a_1,a_4"), CSE7_i_i_i_ap2_ap2("i_2,i_3,i_1;a_2,a_4"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2");
  I2_i_i_ap2_ap2 = ArrayToT();  // release
  I_i_μ̃("i_1,μ̃_19664") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_1,μ̃_19664;a_3"), t_ap1_i("i_1;a_3"), "i_1,μ̃_19664")("i_1,μ̃_19664");
  I_i_i_Κ("i_1,i_3,Κ_1") = TA::einsum(I_i_μ̃("i_1,μ̃_19664"), g_i_μ̃_Κ("i_3,μ̃_19664,Κ_1"), "i_1,i_3,Κ_1")("i_1,i_3,Κ_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  CSE10_i_i_μ̃_μ̃("i_3,i_1,μ̃_19665,μ̃_19666") = TA::einsum(I_i_i_Κ("i_1,i_3,Κ_1"), g_μ̃_μ̃_Κ("μ̃_19665,μ̃_19666,Κ_1"), "i_3,i_1,μ̃_19665,μ̃_19666")("i_3,i_1,μ̃_19665,μ̃_19666");
  I_i_i_Κ = TA::TSpArrayD();  // release
  CSE11_i_i_ap2_μ̃("i_2,i_3,μ̃_19666;a_5") = TA::einsum(C_μ̃_ap2("i_2,i_3,μ̃_19666;a_4"), t_ap2_ap2_i_i("i_2,i_3;a_4,a_5"), "i_2,i_3,μ̃_19666;a_5")("i_2,i_3,μ̃_19666;a_5");
  I2_i_i_ap2_μ̃("i_1,i_2,i_3,μ̃_19666;a_1") = TA::einsum(CSE11_i_i_ap2_μ̃("i_2,i_3,μ̃_19666;a_5"), CSE4_i_i_i_ap2_ap2("i_3,i_1,i_2;a_1,a_5"), "i_1,i_2,i_3,μ̃_19666;a_1")("i_1,i_2,i_3,μ̃_19666;a_1");
  I_i_i_ap2_μ̃("i_1,i_2,μ̃_19665;a_1") = TA::einsum(I2_i_i_ap2_μ̃("i_1,i_2,i_3,μ̃_19666;a_1"), CSE10_i_i_μ̃_μ̃("i_3,i_1,μ̃_19665,μ̃_19666"), "i_1,i_2,μ̃_19665;a_1")("i_1,i_2,μ̃_19665;a_1");
  I2_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,μ̃_19665;a_1"), C_ap2_μ̃("i_1,i_2,μ̃_19665;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-2);
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_μ̃("i_1,μ̃_19679") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_1,μ̃_19679;a_3"), t_ap1_i("i_1;a_3"), "i_1,μ̃_19679")("i_1,μ̃_19679");
  I_i_i_Κ("i_1,i_3,Κ_1") = TA::einsum(I_i_μ̃("i_1,μ̃_19679"), g_i_μ̃_Κ("i_3,μ̃_19679,Κ_1"), "i_1,i_3,Κ_1")("i_1,i_3,Κ_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_2,μ̃_19680") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19680;a_4"), t_ap1_i("i_2;a_4"), "i_2,μ̃_19680")("i_2,μ̃_19680");
  I2_i_i_Κ("i_2,i_4,Κ_1") = TA::einsum(I_i_μ̃("i_2,μ̃_19680"), g_i_μ̃_Κ("i_4,μ̃_19680,Κ_1"), "i_2,i_4,Κ_1")("i_2,i_4,Κ_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  CSE12_i_i_i_i("i_4,i_3,i_2,i_1") = TA::einsum(I_i_i_Κ("i_1,i_3,Κ_1"), I2_i_i_Κ("i_2,i_4,Κ_1"), "i_4,i_3,i_2,i_1")("i_4,i_3,i_2,i_1");
  I2_i_i_Κ = TA::TSpArrayD();  // release
  I_i_i_Κ = TA::TSpArrayD();  // release
  I_i_i_ap2_μ̃("i_3,i_4,μ̃_19682;a_6") = TA::einsum(C_μ̃_ap2("i_3,i_4,μ̃_19682;a_5"), t_ap2_ap2_i_i("i_3,i_4;a_5,a_6"), "i_3,i_4,μ̃_19682;a_6")("i_3,i_4,μ̃_19682;a_6");
  CSE13_i_i_μ̃_μ̃("i_4,i_3,μ̃_19684,μ̃_19682") = TA::einsum<TA::DeNest::True>(I_i_i_ap2_μ̃("i_3,i_4,μ̃_19682;a_6"), C_μ̃_ap2("i_3,i_4,μ̃_19684;a_6"), "i_4,i_3,μ̃_19684,μ̃_19682")("i_4,i_3,μ̃_19684,μ̃_19682");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  CSE14_i_i_ap2_μ̃("i_2,i_1,μ̃_19682;a_1") = TA::einsum(s_μ̃_μ̃("μ̃_19681,μ̃_19682"), C_ap2_μ̃("i_1,i_2,μ̃_19681;a_1"), "i_2,i_1,μ̃_19682;a_1")("i_2,i_1,μ̃_19682;a_1");
  I_i_i_μ̃_μ̃("i_1,i_2,μ̃_19682,μ̃_19684") = TA::einsum(CSE12_i_i_i_i("i_4,i_3,i_2,i_1"), CSE13_i_i_μ̃_μ̃("i_4,i_3,μ̃_19684,μ̃_19682"), "i_1,i_2,μ̃_19682,μ̃_19684")("i_1,i_2,μ̃_19682,μ̃_19684");
  I_i_i_ap2_μ̃("i_1,i_2,μ̃_19684;a_1") = TA::einsum(I_i_i_μ̃_μ̃("i_1,i_2,μ̃_19682,μ̃_19684"), CSE14_i_i_ap2_μ̃("i_2,i_1,μ̃_19682;a_1"), "i_1,i_2,μ̃_19684;a_1")("i_1,i_2,μ̃_19684;a_1");
  I_i_i_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,μ̃_19684;a_1"), CSE14_i_i_ap2_μ̃("i_2,i_1,μ̃_19684;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  CSE15_i_i_i_i("i_4,i_3,i_2,i_1") = TA::einsum(g_i_i_Κ("i_3,i_1,Κ_1"), g_i_i_Κ("i_4,i_2,Κ_1"), "i_4,i_3,i_2,i_1")("i_4,i_3,i_2,i_1");
  I_i_i_μ̃_μ̃("i_1,i_2,μ̃_19728,μ̃_19730") = TA::einsum(CSE15_i_i_i_i("i_4,i_3,i_2,i_1"), CSE13_i_i_μ̃_μ̃("i_4,i_3,μ̃_19730,μ̃_19728"), "i_1,i_2,μ̃_19728,μ̃_19730")("i_1,i_2,μ̃_19728,μ̃_19730");
  I_i_i_ap2_μ̃("i_1,i_2,μ̃_19730;a_1") = TA::einsum(I_i_i_μ̃_μ̃("i_1,i_2,μ̃_19728,μ̃_19730"), CSE14_i_i_ap2_μ̃("i_2,i_1,μ̃_19728;a_1"), "i_1,i_2,μ̃_19730;a_1")("i_1,i_2,μ̃_19730;a_1");
  I_i_i_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,μ̃_19730;a_1"), CSE14_i_i_ap2_μ̃("i_2,i_1,μ̃_19730;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_i_i_ap2("i_1,i_2,i_4;a_1") = TA::einsum(CSE15_i_i_i_i("i_4,i_3,i_2,i_1"), CSE6_i_i_i_ap2("i_2,i_1,i_3;a_1"), "i_1,i_2,i_4;a_1")("i_1,i_2,i_4;a_1");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += TA::einsum(I_i_i_i_ap2("i_1,i_2,i_4;a_1"), CSE6_i_i_i_ap2("i_2,i_1,i_4;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2");
  I_i_i_i_ap2 = ArrayToT();  // release
  I_i_i_ap2_μ̃("i_1,i_2,μ̃_19781;a_4") = TA::einsum(C_μ̃_ap2("i_1,i_2,μ̃_19781;a_3"), t_ap2_ap2_i_i("i_1,i_2;a_3,a_4"), "i_1,i_2,μ̃_19781;a_4")("i_1,i_2,μ̃_19781;a_4");
  I_i_i_μ̃_μ̃("i_1,i_2,μ̃_19781,μ̃_19782") = TA::einsum<TA::DeNest::True>(I_i_i_ap2_μ̃("i_1,i_2,μ̃_19781;a_4"), C_μ̃_ap2("i_1,i_2,μ̃_19782;a_4"), "i_1,i_2,μ̃_19781,μ̃_19782")("i_1,i_2,μ̃_19781,μ̃_19782");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I2_i_i_μ̃_μ̃("i_3,i_4,μ̃_19781,μ̃_19782") = TA::einsum(g_i_μ̃_Κ("i_3,μ̃_19781,Κ_1"), g_i_μ̃_Κ("i_4,μ̃_19782,Κ_1"), "i_3,i_4,μ̃_19781,μ̃_19782")("i_3,i_4,μ̃_19781,μ̃_19782");
  CSE16_i_i_i_i("i_4,i_3,i_2,i_1") = TA::einsum(I_i_i_μ̃_μ̃("i_1,i_2,μ̃_19781,μ̃_19782"), I2_i_i_μ̃_μ̃("i_3,i_4,μ̃_19781,μ̃_19782"), "i_4,i_3,i_2,i_1")("i_4,i_3,i_2,i_1");
  I2_i_i_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_i_i_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_i_i_μ̃_μ̃("i_1,i_2,μ̃_19784,μ̃_19786") = TA::einsum(CSE16_i_i_i_i("i_4,i_3,i_2,i_1"), CSE13_i_i_μ̃_μ̃("i_4,i_3,μ̃_19786,μ̃_19784"), "i_1,i_2,μ̃_19784,μ̃_19786")("i_1,i_2,μ̃_19784,μ̃_19786");
  I_i_i_ap2_μ̃("i_1,i_2,μ̃_19786;a_1") = TA::einsum(I_i_i_μ̃_μ̃("i_1,i_2,μ̃_19784,μ̃_19786"), CSE14_i_i_ap2_μ̃("i_2,i_1,μ̃_19784;a_1"), "i_1,i_2,μ̃_19786;a_1")("i_1,i_2,μ̃_19786;a_1");
  I_i_i_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,μ̃_19786;a_1"), CSE14_i_i_ap2_μ̃("i_2,i_1,μ̃_19786;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_i_i_ap2("i_1,i_2,i_4;a_1") = TA::einsum(CSE16_i_i_i_i("i_4,i_3,i_2,i_1"), CSE6_i_i_i_ap2("i_2,i_1,i_3;a_1"), "i_1,i_2,i_4;a_1")("i_1,i_2,i_4;a_1");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += TA::einsum(I_i_i_i_ap2("i_1,i_2,i_4;a_1"), CSE6_i_i_i_ap2("i_2,i_1,i_4;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2");
  I_i_i_i_ap2 = ArrayToT();  // release
  I_i_μ̃_μ̃_μ̃("i_3,μ̃_19871,μ̃_19872,μ̃_19873") = TA::einsum(g_i_μ̃_Κ("i_3,μ̃_19871,Κ_1"), g_μ̃_μ̃_Κ("μ̃_19872,μ̃_19873,Κ_1"), "i_3,μ̃_19871,μ̃_19872,μ̃_19873")("i_3,μ̃_19871,μ̃_19872,μ̃_19873");
  I_i_i_i_μ̃("i_1,i_2,i_3,μ̃_19872") = TA::einsum(I_i_μ̃_μ̃_μ̃("i_3,μ̃_19871,μ̃_19872,μ̃_19873"), CSE13_i_i_μ̃_μ̃("i_1,i_2,μ̃_19873,μ̃_19871"), "i_1,i_2,i_3,μ̃_19872")("i_1,i_2,i_3,μ̃_19872");
  I_i_μ̃_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_i_i_i_ap2("i_1,i_2,i_3;a_1") = TA::einsum(I_i_i_i_μ̃("i_1,i_2,i_3,μ̃_19872"), C_ap2_μ̃("i_1,i_2,μ̃_19872;a_1"), "i_1,i_2,i_3;a_1")("i_1,i_2,i_3;a_1");
  I_i_i_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_i_ap2("i_1,i_2,i_3;a_1"), CSE6_i_i_i_ap2("i_2,i_1,i_3;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-2);
  I_i_i_i_ap2 = ArrayToT();  // release
  I_i_i_i_μ̃("i_1,i_3,i_4,μ̃_19918") = TA::einsum(g_i_i_Κ("i_3,i_1,Κ_1"), g_i_μ̃_Κ("i_4,μ̃_19918,Κ_1"), "i_1,i_3,i_4,μ̃_19918")("i_1,i_3,i_4,μ̃_19918");
  I_i_μ̃("i_2,μ̃_19918") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19918;a_3"), t_ap1_i("i_2;a_3"), "i_2,μ̃_19918")("i_2,μ̃_19918");
  CSE17_i_i_i_i("i_3,i_4,i_1,i_2") = TA::einsum(I_i_i_i_μ̃("i_1,i_3,i_4,μ̃_19918"), I_i_μ̃("i_2,μ̃_19918"), "i_3,i_4,i_1,i_2")("i_3,i_4,i_1,i_2");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_μ̃_μ̃("i_1,i_2,μ̃_19920,μ̃_19922") = TA::einsum(CSE17_i_i_i_i("i_3,i_4,i_1,i_2"), CSE13_i_i_μ̃_μ̃("i_4,i_3,μ̃_19922,μ̃_19920"), "i_1,i_2,μ̃_19920,μ̃_19922")("i_1,i_2,μ̃_19920,μ̃_19922");
  I_i_i_ap2_μ̃("i_1,i_2,μ̃_19922;a_1") = TA::einsum(I_i_i_μ̃_μ̃("i_1,i_2,μ̃_19920,μ̃_19922"), CSE14_i_i_ap2_μ̃("i_2,i_1,μ̃_19920;a_1"), "i_1,i_2,μ̃_19922;a_1")("i_1,i_2,μ̃_19922;a_1");
  I_i_i_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,μ̃_19922;a_1"), CSE14_i_i_ap2_μ̃("i_2,i_1,μ̃_19922;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (2);
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_i_i_ap2("i_1,i_2,i_4;a_1") = TA::einsum(CSE12_i_i_i_i("i_4,i_3,i_2,i_1"), CSE6_i_i_i_ap2("i_2,i_1,i_3;a_1"), "i_1,i_2,i_4;a_1")("i_1,i_2,i_4;a_1");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += TA::einsum(I_i_i_i_ap2("i_1,i_2,i_4;a_1"), CSE6_i_i_i_ap2("i_2,i_1,i_4;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2");
  I_i_i_i_ap2 = ArrayToT();  // release
  I_i_ap2_Κ("i_1,i_3,Κ_1;a_3") = TA::einsum(g_i_μ̃_Κ("i_3,μ̃_19707,Κ_1"), C_μ̃_ap2("i_1,i_3,μ̃_19707;a_3"), "i_1,i_3,Κ_1;a_3")("i_1,i_3,Κ_1;a_3");
  I_i_i_ap2_μ̃("i_1,i_3,μ̃_19711;a_3") = TA::einsum(C_μ̃_ap2("i_1,i_3,μ̃_19711;a_5"), t_ap2_ap2_i_i("i_3,i_1;a_3,a_5"), "i_1,i_3,μ̃_19711;a_3")("i_1,i_3,μ̃_19711;a_3");
  CSE18_i_μ̃_Κ("i_1,μ̃_19711,Κ_1") = TA::einsum<TA::DeNest::True>(I_i_ap2_Κ("i_1,i_3,Κ_1;a_3"), I_i_i_ap2_μ̃("i_1,i_3,μ̃_19711;a_3"), "i_1,μ̃_19711,Κ_1")("i_1,μ̃_19711,Κ_1");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_ap2_Κ = ArrayToT();  // release
  I_i_μ̃("i_2,μ̃_19709") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19709;a_4"), t_ap1_i("i_2;a_4"), "i_2,μ̃_19709")("i_2,μ̃_19709");
  I_i_μ̃_Κ("i_2,μ̃_19708,Κ_1") = TA::einsum(I_i_μ̃("i_2,μ̃_19709"), g_μ̃_μ̃_Κ("μ̃_19708,μ̃_19709,Κ_1"), "i_2,μ̃_19708,Κ_1")("i_2,μ̃_19708,Κ_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  CSE19_i_i_ap2_Κ("i_1,i_2,Κ_1;a_2") = TA::einsum(I_i_μ̃_Κ("i_2,μ̃_19708,Κ_1"), C_ap2_μ̃("i_1,i_2,μ̃_19708;a_2"), "i_1,i_2,Κ_1;a_2")("i_1,i_2,Κ_1;a_2");
  I_i_μ̃_Κ = TA::TSpArrayD();  // release
  I_i_i_ap2_μ̃("i_1,i_2,μ̃_19711;a_2") = TA::einsum(CSE18_i_μ̃_Κ("i_1,μ̃_19711,Κ_1"), CSE19_i_i_ap2_Κ("i_1,i_2,Κ_1;a_2"), "i_1,i_2,μ̃_19711;a_2")("i_1,i_2,μ̃_19711;a_2");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,μ̃_19711;a_2"), CSE14_i_i_ap2_μ̃("i_2,i_1,μ̃_19711;a_1"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (4);
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_ap2_Κ("i_1,i_3,Κ_1;a_3") = TA::einsum(g_i_μ̃_Κ("i_3,μ̃_19721,Κ_1"), C_μ̃_ap2("i_1,i_3,μ̃_19721;a_3"), "i_1,i_3,Κ_1;a_3")("i_1,i_3,Κ_1;a_3");
  I_i_i_ap2_μ̃("i_1,i_3,μ̃_19724;a_3") = TA::einsum(C_μ̃_ap2("i_1,i_3,μ̃_19724;a_5"), t_ap2_ap2_i_i("i_3,i_1;a_3,a_5"), "i_1,i_3,μ̃_19724;a_3")("i_1,i_3,μ̃_19724;a_3");
  I_i_μ̃_Κ("i_1,μ̃_19724,Κ_1") = TA::einsum<TA::DeNest::True>(I_i_ap2_Κ("i_1,i_3,Κ_1;a_3"), I_i_i_ap2_μ̃("i_1,i_3,μ̃_19724;a_3"), "i_1,μ̃_19724,Κ_1")("i_1,μ̃_19724,Κ_1");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_ap2_Κ = ArrayToT();  // release
  I_ap2_μ̃("i_1,i_2,μ̃_19724;a_1") = TA::einsum(s_μ̃_μ̃("μ̃_19723,μ̃_19724"), C_ap2_μ̃("i_1,i_2,μ̃_19723;a_1"), "i_1,i_2,μ̃_19724;a_1")("i_1,i_2,μ̃_19724;a_1");
  CSE20_i_i_ap2_Κ("i_2,i_1,Κ_1;a_1") = TA::einsum(I_i_μ̃_Κ("i_1,μ̃_19724,Κ_1"), I_ap2_μ̃("i_1,i_2,μ̃_19724;a_1"), "i_2,i_1,Κ_1;a_1")("i_2,i_1,Κ_1;a_1");
  I_ap2_μ̃ = ArrayToT();  // release
  I_i_μ̃_Κ = TA::TSpArrayD();  // release
  I_i_i_ap2_μ̃("i_1,i_2,μ̃_19726;a_1") = TA::einsum(CSE20_i_i_ap2_Κ("i_2,i_1,Κ_1;a_1"), CSE18_i_μ̃_Κ("i_2,μ̃_19726,Κ_1"), "i_1,i_2,μ̃_19726;a_1")("i_1,i_2,μ̃_19726;a_1");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,μ̃_19726;a_1"), CSE14_i_i_ap2_μ̃("i_2,i_1,μ̃_19726;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (4);
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_ap2_Κ("i_2,i_4,Κ_1;a_4") = TA::einsum(g_i_μ̃_Κ("i_4,μ̃_19767,Κ_1"), C_μ̃_ap2("i_2,i_4,μ̃_19767;a_4"), "i_2,i_4,Κ_1;a_4")("i_2,i_4,Κ_1;a_4");
  I_i_i_ap2_μ̃("i_2,i_4,μ̃_19771;a_4") = TA::einsum(C_μ̃_ap2("i_2,i_4,μ̃_19771;a_6"), t_ap2_ap2_i_i("i_2,i_4;a_4,a_6"), "i_2,i_4,μ̃_19771;a_4")("i_2,i_4,μ̃_19771;a_4");
  CSE21_i_μ̃_Κ("i_2,μ̃_19771,Κ_1") = TA::einsum<TA::DeNest::True>(I_i_ap2_Κ("i_2,i_4,Κ_1;a_4"), I_i_i_ap2_μ̃("i_2,i_4,μ̃_19771;a_4"), "i_2,μ̃_19771,Κ_1")("i_2,μ̃_19771,Κ_1");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_ap2_Κ = ArrayToT();  // release
  I_i_i_ap2_μ̃("i_1,i_2,μ̃_19771;a_1") = TA::einsum(CSE20_i_i_ap2_Κ("i_2,i_1,Κ_1;a_1"), CSE21_i_μ̃_Κ("i_2,μ̃_19771,Κ_1"), "i_1,i_2,μ̃_19771;a_1")("i_1,i_2,μ̃_19771;a_1");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,μ̃_19771;a_1"), CSE14_i_i_ap2_μ̃("i_2,i_1,μ̃_19771;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-4);
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_ap2_Κ("i_2,i_1,Κ_1;a_1") = TA::einsum(CSE21_i_μ̃_Κ("i_1,μ̃_19822,Κ_1"), CSE14_i_i_ap2_μ̃("i_2,i_1,μ̃_19822;a_1"), "i_2,i_1,Κ_1;a_1")("i_2,i_1,Κ_1;a_1");
  I_i_i_ap2_μ̃("i_1,i_2,μ̃_19824;a_1") = TA::einsum(I_i_ap2_Κ("i_2,i_1,Κ_1;a_1"), CSE21_i_μ̃_Κ("i_2,μ̃_19824,Κ_1"), "i_1,i_2,μ̃_19824;a_1")("i_1,i_2,μ̃_19824;a_1");
  I_i_ap2_Κ = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,μ̃_19824;a_1"), CSE14_i_i_ap2_μ̃("i_2,i_1,μ̃_19824;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_i_ap2_μ̃("i_1,i_2,μ̃_19904;a_2") = TA::einsum(CSE21_i_μ̃_Κ("i_1,μ̃_19904,Κ_1"), CSE19_i_i_ap2_Κ("i_1,i_2,Κ_1;a_2"), "i_1,i_2,μ̃_19904;a_2")("i_1,i_2,μ̃_19904;a_2");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,μ̃_19904;a_2"), CSE14_i_i_ap2_μ̃("i_2,i_1,μ̃_19904;a_1"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-2);
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  CSE22_i_i_ap2_Κ("i_3,i_4,Κ_1;a_4") = TA::einsum(g_i_μ̃_Κ("i_4,μ̃_19732,Κ_1"), C_μ̃_ap2("i_3,i_4,μ̃_19732;a_4"), "i_3,i_4,Κ_1;a_4")("i_3,i_4,Κ_1;a_4");
  I_i_i_ap2_μ̃("i_3,i_4,μ̃_19731;a_4") = TA::einsum(g_i_μ̃_Κ("i_3,μ̃_19731,Κ_1"), CSE22_i_i_ap2_Κ("i_3,i_4,Κ_1;a_4"), "i_3,i_4,μ̃_19731;a_4")("i_3,i_4,μ̃_19731;a_4");
  I2_ap2_μ̃("i_3,i_4,μ̃_19731;a_5") = TA::einsum(I_i_i_ap2_μ̃("i_3,i_4,μ̃_19731;a_4"), t_ap2_ap2_i_i("i_3,i_4;a_4,a_5"), "i_3,i_4,μ̃_19731;a_5")("i_3,i_4,μ̃_19731;a_5");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I2_μ̃_μ̃("μ̃_19731,μ̃_19734") = TA::einsum<TA::DeNest::True>(I2_ap2_μ̃("i_3,i_4,μ̃_19731;a_5"), C_μ̃_ap2("i_3,i_4,μ̃_19734;a_5"), "μ̃_19731,μ̃_19734")("μ̃_19731,μ̃_19734");
  I2_ap2_μ̃ = ArrayToT();  // release
  I_μ̃_μ̃("μ̃_19731,μ̃_19733") = TA::einsum(I2_μ̃_μ̃("μ̃_19731,μ̃_19734"), s_μ̃_μ̃("μ̃_19733,μ̃_19734"), "μ̃_19731,μ̃_19733")("μ̃_19731,μ̃_19733");
  I2_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_ap2_μ̃("i_1,i_2,μ̃_19733;a_3") = TA::einsum(I_μ̃_μ̃("μ̃_19731,μ̃_19733"), C_μ̃_ap2("i_1,i_2,μ̃_19731;a_3"), "i_1,i_2,μ̃_19733;a_3")("i_1,i_2,μ̃_19733;a_3");
  I_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_ap2_ap2("i_1,i_2;a_2,a_3") = TA::einsum(I_ap2_μ̃("i_1,i_2,μ̃_19733;a_3"), C_ap2_μ̃("i_1,i_2,μ̃_19733;a_2"), "i_1,i_2;a_2,a_3")("i_1,i_2;a_2,a_3");
  I_ap2_μ̃ = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_ap2_ap2("i_1,i_2;a_2,a_3"), t_ap2_ap2_i_i("i_1,i_2;a_1,a_3"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (2);
  I_ap2_ap2 = ArrayToT();  // release
  I_i_i_ap2_μ̃("i_3,i_4,μ̃_19778;a_3") = TA::einsum(CSE22_i_i_ap2_Κ("i_4,i_3,Κ_1;a_3"), g_i_μ̃_Κ("i_4,μ̃_19778,Κ_1"), "i_3,i_4,μ̃_19778;a_3")("i_3,i_4,μ̃_19778;a_3");
  I2_ap2_μ̃("i_3,i_4,μ̃_19778;a_5") = TA::einsum(I_i_i_ap2_μ̃("i_3,i_4,μ̃_19778;a_3"), t_ap2_ap2_i_i("i_3,i_4;a_3,a_5"), "i_3,i_4,μ̃_19778;a_5")("i_3,i_4,μ̃_19778;a_5");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I2_μ̃_μ̃("μ̃_19778,μ̃_19780") = TA::einsum<TA::DeNest::True>(I2_ap2_μ̃("i_3,i_4,μ̃_19778;a_5"), C_μ̃_ap2("i_3,i_4,μ̃_19780;a_5"), "μ̃_19778,μ̃_19780")("μ̃_19778,μ̃_19780");
  I2_ap2_μ̃ = ArrayToT();  // release
  I_μ̃_μ̃("μ̃_19778,μ̃_19779") = TA::einsum(I2_μ̃_μ̃("μ̃_19778,μ̃_19780"), s_μ̃_μ̃("μ̃_19779,μ̃_19780"), "μ̃_19778,μ̃_19779")("μ̃_19778,μ̃_19779");
  I2_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_ap2_μ̃("i_1,i_2,μ̃_19779;a_4") = TA::einsum(I_μ̃_μ̃("μ̃_19778,μ̃_19779"), C_μ̃_ap2("i_1,i_2,μ̃_19778;a_4"), "i_1,i_2,μ̃_19779;a_4")("i_1,i_2,μ̃_19779;a_4");
  I_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_ap2_ap2("i_1,i_2;a_2,a_4") = TA::einsum(I_ap2_μ̃("i_1,i_2,μ̃_19779;a_4"), C_ap2_μ̃("i_1,i_2,μ̃_19779;a_2"), "i_1,i_2;a_2,a_4")("i_1,i_2;a_2,a_4");
  I_ap2_μ̃ = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_ap2_ap2("i_1,i_2;a_2,a_4"), t_ap2_ap2_i_i("i_1,i_2;a_1,a_4"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-4);
  I_ap2_ap2 = ArrayToT();  // release
  I_i_ap2_Κ("i_2,i_3,Κ_1;a_3") = TA::einsum(g_i_μ̃_Κ("i_3,μ̃_19792,Κ_1"), C_μ̃_ap2("i_2,i_3,μ̃_19792;a_3"), "i_2,i_3,Κ_1;a_3")("i_2,i_3,Κ_1;a_3");
  I2_i_ap2_Κ("i_2,i_1,Κ_1;a_1") = TA::einsum(g_μ̃_i_Κ("μ̃_19793,i_1,Κ_1"), C_ap2_μ̃("i_1,i_2,μ̃_19793;a_1"), "i_2,i_1,Κ_1;a_1")("i_2,i_1,Κ_1;a_1");
  CSE23_i_i_i_ap2_ap2("i_2,i_3,i_1;a_1,a_3") = TA::einsum(I_i_ap2_Κ("i_2,i_3,Κ_1;a_3"), I2_i_ap2_Κ("i_2,i_1,Κ_1;a_1"), "i_2,i_3,i_1;a_1,a_3")("i_2,i_3,i_1;a_1,a_3");
  I2_i_ap2_Κ = ArrayToT();  // release
  I_i_ap2_Κ = ArrayToT();  // release
  I2_i_i_ap2_ap2("i_3,i_1,i_2;a_1,a_4") = TA::einsum(CSE23_i_i_i_ap2_ap2("i_2,i_3,i_1;a_1,a_3"), t_ap2_ap2_i_i("i_2,i_3;a_3,a_4"), "i_3,i_1,i_2;a_1,a_4")("i_3,i_1,i_2;a_1,a_4");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I2_i_i_ap2_ap2("i_3,i_1,i_2;a_1,a_4"), CSE4_i_i_i_ap2_ap2("i_3,i_1,i_2;a_2,a_4"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-2);
  I2_i_i_ap2_ap2 = ArrayToT();  // release
  I2_i_i_ap2_ap2("i_3,i_1,i_2;a_1,a_4") = TA::einsum(CSE23_i_i_i_ap2_ap2("i_2,i_3,i_1;a_1,a_3"), t_ap2_ap2_i_i("i_3,i_2;a_3,a_4"), "i_3,i_1,i_2;a_1,a_4")("i_3,i_1,i_2;a_1,a_4");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I2_i_i_ap2_ap2("i_3,i_1,i_2;a_1,a_4"), CSE4_i_i_i_ap2_ap2("i_3,i_1,i_2;a_2,a_4"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (4);
  I2_i_i_ap2_ap2 = ArrayToT();  // release
  I2_i_i_ap2_μ̃("i_1,i_2,i_3,μ̃_19843;a_2") = TA::einsum(CSE11_i_i_ap2_μ̃("i_3,i_2,μ̃_19843;a_5"), CSE4_i_i_i_ap2_ap2("i_3,i_1,i_2;a_2,a_5"), "i_1,i_2,i_3,μ̃_19843;a_2")("i_1,i_2,i_3,μ̃_19843;a_2");
  I_i_i_ap2_μ̃("i_1,i_2,μ̃_19842;a_2") = TA::einsum(I2_i_i_ap2_μ̃("i_1,i_2,i_3,μ̃_19843;a_2"), CSE10_i_i_μ̃_μ̃("i_3,i_1,μ̃_19842,μ̃_19843"), "i_1,i_2,μ̃_19842;a_2")("i_1,i_2,μ̃_19842;a_2");
  I2_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,μ̃_19842;a_2"), C_ap2_μ̃("i_1,i_2,μ̃_19842;a_1"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-2);
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_μ̃("i_2,μ̃_19685") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19685;a_3"), t_ap1_i("i_2;a_3"), "i_2,μ̃_19685")("i_2,μ̃_19685");
  I_i_i_Κ("i_2,i_3,Κ_1") = TA::einsum(I_i_μ̃("i_2,μ̃_19685"), g_i_μ̃_Κ("i_3,μ̃_19685,Κ_1"), "i_2,i_3,Κ_1")("i_2,i_3,Κ_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_i_μ̃("i_2,i_3,i_4,μ̃_19686") = TA::einsum(I_i_i_Κ("i_2,i_3,Κ_1"), g_i_μ̃_Κ("i_4,μ̃_19686,Κ_1"), "i_2,i_3,i_4,μ̃_19686")("i_2,i_3,i_4,μ̃_19686");
  I_i_i_Κ = TA::TSpArrayD();  // release
  CSE24_i_i_i_i_ap2("i_1,i_3,i_4,i_2;a_4") = TA::einsum(I_i_i_i_μ̃("i_2,i_3,i_4,μ̃_19686"), C_μ̃_ap2("i_1,i_4,μ̃_19686;a_4"), "i_1,i_3,i_4,i_2;a_4")("i_1,i_3,i_4,i_2;a_4");
  I_i_i_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_i_i_ap2("i_4,i_1,i_2,i_3;a_5") = TA::einsum(CSE24_i_i_i_i_ap2("i_1,i_3,i_4,i_2;a_4"), t_ap2_ap2_i_i("i_1,i_4;a_4,a_5"), "i_4,i_1,i_2,i_3;a_5")("i_4,i_1,i_2,i_3;a_5");
  I_i_i_i_ap2("i_1,i_2,i_3;a_1") = TA::einsum(I2_i_i_i_ap2("i_4,i_1,i_2,i_3;a_5"), CSE4_i_i_i_ap2_ap2("i_4,i_2,i_1;a_1,a_5"), "i_1,i_2,i_3;a_1")("i_1,i_2,i_3;a_1");
  I2_i_i_i_ap2 = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_i_ap2("i_1,i_2,i_3;a_1"), CSE6_i_i_i_ap2("i_2,i_1,i_3;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (2);
  I_i_i_i_ap2 = ArrayToT();  // release
  I2_i_i_i_ap2("i_4,i_1,i_2,i_3;a_5") = TA::einsum(CSE24_i_i_i_i_ap2("i_1,i_3,i_4,i_2;a_4"), t_ap2_ap2_i_i("i_4,i_1;a_4,a_5"), "i_4,i_1,i_2,i_3;a_5")("i_4,i_1,i_2,i_3;a_5");
  I_i_i_i_ap2("i_1,i_2,i_3;a_1") = TA::einsum(I2_i_i_i_ap2("i_4,i_1,i_2,i_3;a_5"), CSE4_i_i_i_ap2_ap2("i_4,i_2,i_1;a_1,a_5"), "i_1,i_2,i_3;a_1")("i_1,i_2,i_3;a_1");
  I2_i_i_i_ap2 = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_i_ap2("i_1,i_2,i_3;a_1"), CSE6_i_i_i_ap2("i_2,i_1,i_3;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-4);
  I_i_i_i_ap2 = ArrayToT();  // release
  I_i_μ̃("i_2,μ̃_19825") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19825;a_3"), t_ap1_i("i_2;a_3"), "i_2,μ̃_19825")("i_2,μ̃_19825");
  I_i_i_Κ("i_2,i_3,Κ_1") = TA::einsum(I_i_μ̃("i_2,μ̃_19825"), g_i_μ̃_Κ("i_3,μ̃_19825,Κ_1"), "i_2,i_3,Κ_1")("i_2,i_3,Κ_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_i_μ̃("i_2,i_3,i_4,μ̃_19826") = TA::einsum(I_i_i_Κ("i_2,i_3,Κ_1"), g_i_μ̃_Κ("i_4,μ̃_19826,Κ_1"), "i_2,i_3,i_4,μ̃_19826")("i_2,i_3,i_4,μ̃_19826");
  I_i_i_Κ = TA::TSpArrayD();  // release
  CSE25_i_i_i_i_ap2("i_1,i_4,i_3,i_2;a_4") = TA::einsum(I_i_i_i_μ̃("i_2,i_3,i_4,μ̃_19826"), C_μ̃_ap2("i_1,i_3,μ̃_19826;a_4"), "i_1,i_4,i_3,i_2;a_4")("i_1,i_4,i_3,i_2;a_4");
  I_i_i_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_i_i_ap2("i_3,i_1,i_2,i_4;a_6") = TA::einsum(CSE25_i_i_i_i_ap2("i_1,i_4,i_3,i_2;a_4"), t_ap2_ap2_i_i("i_1,i_3;a_4,a_6"), "i_3,i_1,i_2,i_4;a_6")("i_3,i_1,i_2,i_4;a_6");
  I_i_i_i_ap2("i_1,i_2,i_4;a_2") = TA::einsum(I2_i_i_i_ap2("i_3,i_1,i_2,i_4;a_6"), CSE4_i_i_i_ap2_ap2("i_3,i_2,i_1;a_2,a_6"), "i_1,i_2,i_4;a_2")("i_1,i_2,i_4;a_2");
  I2_i_i_i_ap2 = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_i_ap2("i_1,i_2,i_4;a_2"), CSE6_i_i_i_ap2("i_2,i_1,i_4;a_1"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (2);
  I_i_i_i_ap2 = ArrayToT();  // release
  I2_i_i_i_ap2("i_3,i_1,i_2,i_4;a_5") = TA::einsum(CSE25_i_i_i_i_ap2("i_1,i_4,i_3,i_2;a_4"), t_ap2_ap2_i_i("i_3,i_1;a_4,a_5"), "i_3,i_1,i_2,i_4;a_5")("i_3,i_1,i_2,i_4;a_5");
  I_i_i_i_ap2("i_1,i_2,i_4;a_1") = TA::einsum(I2_i_i_i_ap2("i_3,i_1,i_2,i_4;a_5"), CSE4_i_i_i_ap2_ap2("i_3,i_2,i_1;a_1,a_5"), "i_1,i_2,i_4;a_1")("i_1,i_2,i_4;a_1");
  I2_i_i_i_ap2 = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_i_ap2("i_1,i_2,i_4;a_1"), CSE6_i_i_i_ap2("i_2,i_1,i_4;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (2);
  I_i_i_i_ap2 = ArrayToT();  // release
  I_i_μ̃("i_1,μ̃_19753") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_1,μ̃_19753;a_3"), t_ap1_i("i_1;a_3"), "i_1,μ̃_19753")("i_1,μ̃_19753");
  CSE26_i_i_Κ("i_3,i_1,Κ_1") = TA::einsum(I_i_μ̃("i_1,μ̃_19753"), g_i_μ̃_Κ("i_3,μ̃_19753,Κ_1"), "i_3,i_1,Κ_1")("i_3,i_1,Κ_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  CSE27_i_i_ap2_Κ("i_1,i_2,Κ_1;a_2") = TA::einsum(g_μ̃_i_Κ("μ̃_19754,i_2,Κ_1"), C_ap2_μ̃("i_1,i_2,μ̃_19754;a_2"), "i_1,i_2,Κ_1;a_2")("i_1,i_2,Κ_1;a_2");
  I_i_i_i_ap2("i_1,i_2,i_3;a_2") = TA::einsum(CSE26_i_i_Κ("i_3,i_1,Κ_1"), CSE27_i_i_ap2_Κ("i_1,i_2,Κ_1;a_2"), "i_1,i_2,i_3;a_2")("i_1,i_2,i_3;a_2");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_i_ap2("i_1,i_2,i_3;a_2"), CSE6_i_i_i_ap2("i_2,i_1,i_3;a_1"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-2);
  I_i_i_i_ap2 = ArrayToT();  // release
  I_i_μ̃("i_2,μ̃_19798") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19798;a_3"), t_ap1_i("i_2;a_3"), "i_2,μ̃_19798")("i_2,μ̃_19798");
  CSE28_i_μ̃_Κ("i_2,μ̃_19797,Κ_1") = TA::einsum(I_i_μ̃("i_2,μ̃_19798"), g_μ̃_μ̃_Κ("μ̃_19797,μ̃_19798,Κ_1"), "i_2,μ̃_19797,Κ_1")("i_2,μ̃_19797,Κ_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_ap2_μ̃("i_1,i_2,μ̃_19797;a_1") = TA::einsum(CSE27_i_i_ap2_Κ("i_2,i_1,Κ_1;a_1"), CSE28_i_μ̃_Κ("i_2,μ̃_19797,Κ_1"), "i_1,i_2,μ̃_19797;a_1")("i_1,i_2,μ̃_19797;a_1");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,μ̃_19797;a_1"), C_ap2_μ̃("i_1,i_2,μ̃_19797;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (2);
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_i_ap2_μ̃("i_1,i_2,μ̃_19805;a_1") = TA::einsum(CSE27_i_i_ap2_Κ("i_2,i_1,Κ_1;a_1"), g_μ̃_i_Κ("μ̃_19805,i_2,Κ_1"), "i_1,i_2,μ̃_19805;a_1")("i_1,i_2,μ̃_19805;a_1");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,μ̃_19805;a_1"), C_ap2_μ̃("i_1,i_2,μ̃_19805;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_μ̃("i_4,μ̃_19876") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_4,μ̃_19876;a_3"), t_ap1_i("i_4;a_3"), "i_4,μ̃_19876")("i_4,μ̃_19876");
  I_i_i_Κ("i_3,i_4,Κ_1") = TA::einsum(I_i_μ̃("i_4,μ̃_19876"), g_i_μ̃_Κ("i_3,μ̃_19876,Κ_1"), "i_3,i_4,Κ_1")("i_3,i_4,Κ_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  CSE29_i_μ̃("i_3,μ̃_19877") = TA::einsum(I_i_i_Κ("i_3,i_4,Κ_1"), g_i_μ̃_Κ("i_4,μ̃_19877,Κ_1"), "i_3,μ̃_19877")("i_3,μ̃_19877");
  I_i_i_Κ = TA::TSpArrayD();  // release
  I_i_ap2("i_1,i_2,i_3;a_4") = TA::einsum(CSE29_i_μ̃("i_3,μ̃_19877"), C_μ̃_ap2("i_1,i_2,μ̃_19877;a_4"), "i_1,i_2,i_3;a_4")("i_1,i_2,i_3;a_4");
  I_i_i_i_ap2("i_1,i_2,i_3;a_1") = TA::einsum(I_i_ap2("i_1,i_2,i_3;a_4"), t_ap2_ap2_i_i("i_1,i_2;a_1,a_4"), "i_1,i_2,i_3;a_1")("i_1,i_2,i_3;a_1");
  I_i_ap2 = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_i_ap2("i_1,i_2,i_3;a_1"), CSE6_i_i_i_ap2("i_2,i_1,i_3;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (2);
  I_i_i_i_ap2 = ArrayToT();  // release
  I_i_i("i_2,i_3") = TA::einsum(CSE29_i_μ̃("i_3,μ̃_19886"), CSE3_i_μ̃("i_2,μ̃_19886"), "i_2,i_3")("i_2,i_3");
  I2_i_i_ap2_ap2("i_1,i_2,i_3;a_2,a_6") = TA::einsum(I_i_i("i_2,i_3"), CSE4_i_i_i_ap2_ap2("i_3,i_2,i_1;a_2,a_6"), "i_1,i_2,i_3;a_2,a_6")("i_1,i_2,i_3;a_2,a_6");
  I_i_i = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I2_i_i_ap2_ap2("i_1,i_2,i_3;a_2,a_6"), CSE5_i_i_i_ap2_ap2("i_2,i_3,i_1;a_1,a_6"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (2);
  I2_i_i_ap2_ap2 = ArrayToT();  // release
  I_i_i_i_μ̃("i_1,i_2,i_3,μ̃_19881") = TA::einsum(CSE26_i_i_Κ("i_3,i_2,Κ_1"), CSE28_i_μ̃_Κ("i_1,μ̃_19881,Κ_1"), "i_1,i_2,i_3,μ̃_19881")("i_1,i_2,i_3,μ̃_19881");
  I_i_i_i_ap2("i_1,i_2,i_3;a_1") = TA::einsum(I_i_i_i_μ̃("i_1,i_2,i_3,μ̃_19881"), C_ap2_μ̃("i_1,i_2,μ̃_19881;a_1"), "i_1,i_2,i_3;a_1")("i_1,i_2,i_3;a_1");
  I_i_i_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_i_ap2("i_1,i_2,i_3;a_1"), CSE6_i_i_i_ap2("i_2,i_1,i_3;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-2);
  I_i_i_i_ap2 = ArrayToT();  // release
  I_i_i_i_μ̃("i_2,i_3,i_4,μ̃_19669") = TA::einsum(g_i_i_Κ("i_3,i_2,Κ_1"), g_i_μ̃_Κ("i_4,μ̃_19669,Κ_1"), "i_2,i_3,i_4,μ̃_19669")("i_2,i_3,i_4,μ̃_19669");
  CSE30_i_i_i_i_ap2("i_1,i_4,i_3,i_2;a_3") = TA::einsum(I_i_i_i_μ̃("i_2,i_3,i_4,μ̃_19669"), C_μ̃_ap2("i_1,i_3,μ̃_19669;a_3"), "i_1,i_4,i_3,i_2;a_3")("i_1,i_4,i_3,i_2;a_3");
  I_i_i_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_i_i_ap2("i_3,i_1,i_2,i_4;a_4") = TA::einsum(CSE30_i_i_i_i_ap2("i_1,i_4,i_3,i_2;a_3"), t_ap2_ap2_i_i("i_1,i_3;a_3,a_4"), "i_3,i_1,i_2,i_4;a_4")("i_3,i_1,i_2,i_4;a_4");
  I_i_i_i_ap2("i_1,i_2,i_4;a_2") = TA::einsum(I2_i_i_i_ap2("i_3,i_1,i_2,i_4;a_4"), CSE4_i_i_i_ap2_ap2("i_3,i_2,i_1;a_2,a_4"), "i_1,i_2,i_4;a_2")("i_1,i_2,i_4;a_2");
  I2_i_i_i_ap2 = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_i_ap2("i_1,i_2,i_4;a_2"), CSE6_i_i_i_ap2("i_2,i_1,i_4;a_1"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (2);
  I_i_i_i_ap2 = ArrayToT();  // release
  I2_i_i_i_ap2("i_3,i_1,i_2,i_4;a_4") = TA::einsum(CSE30_i_i_i_i_ap2("i_1,i_4,i_3,i_2;a_3"), t_ap2_ap2_i_i("i_3,i_1;a_3,a_4"), "i_3,i_1,i_2,i_4;a_4")("i_3,i_1,i_2,i_4;a_4");
  I_i_i_i_ap2("i_1,i_2,i_4;a_1") = TA::einsum(I2_i_i_i_ap2("i_3,i_1,i_2,i_4;a_4"), CSE4_i_i_i_ap2_ap2("i_3,i_2,i_1;a_1,a_4"), "i_1,i_2,i_4;a_1")("i_1,i_2,i_4;a_1");
  I2_i_i_i_ap2 = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_i_ap2("i_1,i_2,i_4;a_1"), CSE6_i_i_i_ap2("i_2,i_1,i_4;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (2);
  I_i_i_i_ap2 = ArrayToT();  // release
  I_i_i_i_ap2("i_1,i_2,i_3;a_1") = TA::einsum(CSE17_i_i_i_i("i_3,i_4,i_1,i_2"), CSE6_i_i_i_ap2("i_2,i_1,i_4;a_1"), "i_1,i_2,i_3;a_1")("i_1,i_2,i_3;a_1");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_i_ap2("i_1,i_2,i_3;a_1"), CSE6_i_i_i_ap2("i_2,i_1,i_3;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (2);
  I_i_i_i_ap2 = ArrayToT();  // release
  I_i_i_i_μ̃("i_2,i_3,i_4,μ̃_19806") = TA::einsum(g_i_i_Κ("i_3,i_2,Κ_1"), g_i_μ̃_Κ("i_4,μ̃_19806,Κ_1"), "i_2,i_3,i_4,μ̃_19806")("i_2,i_3,i_4,μ̃_19806");
  CSE31_i_i_i_i_ap2("i_1,i_3,i_4,i_2;a_3") = TA::einsum(I_i_i_i_μ̃("i_2,i_3,i_4,μ̃_19806"), C_μ̃_ap2("i_1,i_4,μ̃_19806;a_3"), "i_1,i_3,i_4,i_2;a_3")("i_1,i_3,i_4,i_2;a_3");
  I_i_i_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_i_i_ap2("i_4,i_1,i_2,i_3;a_4") = TA::einsum(CSE31_i_i_i_i_ap2("i_1,i_3,i_4,i_2;a_3"), t_ap2_ap2_i_i("i_4,i_1;a_3,a_4"), "i_4,i_1,i_2,i_3;a_4")("i_4,i_1,i_2,i_3;a_4");
  I_i_i_i_ap2("i_1,i_2,i_3;a_1") = TA::einsum(I2_i_i_i_ap2("i_4,i_1,i_2,i_3;a_4"), CSE4_i_i_i_ap2_ap2("i_4,i_2,i_1;a_1,a_4"), "i_1,i_2,i_3;a_1")("i_1,i_2,i_3;a_1");
  I2_i_i_i_ap2 = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_i_ap2("i_1,i_2,i_3;a_1"), CSE6_i_i_i_ap2("i_2,i_1,i_3;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-4);
  I_i_i_i_ap2 = ArrayToT();  // release
  I2_i_i_i_ap2("i_4,i_1,i_2,i_3;a_4") = TA::einsum(CSE31_i_i_i_i_ap2("i_1,i_3,i_4,i_2;a_3"), t_ap2_ap2_i_i("i_1,i_4;a_3,a_4"), "i_4,i_1,i_2,i_3;a_4")("i_4,i_1,i_2,i_3;a_4");
  I_i_i_i_ap2("i_1,i_2,i_3;a_1") = TA::einsum(I2_i_i_i_ap2("i_4,i_1,i_2,i_3;a_4"), CSE4_i_i_i_ap2_ap2("i_4,i_2,i_1;a_1,a_4"), "i_1,i_2,i_3;a_1")("i_1,i_2,i_3;a_1");
  I2_i_i_i_ap2 = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_i_ap2("i_1,i_2,i_3;a_1"), CSE6_i_i_i_ap2("i_2,i_1,i_3;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (2);
  I_i_i_i_ap2 = ArrayToT();  // release
  CSE32_i_i_μ̃_μ̃("i_3,i_2,μ̃_19857,μ̃_19858") = TA::einsum(g_i_i_Κ("i_3,i_2,Κ_1"), g_μ̃_μ̃_Κ("μ̃_19857,μ̃_19858,Κ_1"), "i_3,i_2,μ̃_19857,μ̃_19858")("i_3,i_2,μ̃_19857,μ̃_19858");
  I_i_i_i_μ̃("i_1,i_2,i_3,μ̃_19857") = TA::einsum(CSE32_i_i_μ̃_μ̃("i_3,i_2,μ̃_19857,μ̃_19858"), CSE3_i_μ̃("i_1,μ̃_19858"), "i_1,i_2,i_3,μ̃_19857")("i_1,i_2,i_3,μ̃_19857");
  I_i_i_i_ap2("i_1,i_2,i_3;a_1") = TA::einsum(I_i_i_i_μ̃("i_1,i_2,i_3,μ̃_19857"), C_ap2_μ̃("i_1,i_2,μ̃_19857;a_1"), "i_1,i_2,i_3;a_1")("i_1,i_2,i_3;a_1");
  I_i_i_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_i_ap2("i_1,i_2,i_3;a_1"), CSE6_i_i_i_ap2("i_2,i_1,i_3;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-2);
  I_i_i_i_ap2 = ArrayToT();  // release
  I_i_ap2("i_1,i_2,i_3;a_3") = TA::einsum(f_i_μ̃("i_3,μ̃_19891"), C_μ̃_ap2("i_1,i_2,μ̃_19891;a_3"), "i_1,i_2,i_3;a_3")("i_1,i_2,i_3;a_3");
  I_i_i_i_ap2("i_1,i_2,i_3;a_1") = TA::einsum(I_i_ap2("i_1,i_2,i_3;a_3"), t_ap2_ap2_i_i("i_1,i_2;a_1,a_3"), "i_1,i_2,i_3;a_1")("i_1,i_2,i_3;a_1");
  I_i_ap2 = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_i_ap2("i_1,i_2,i_3;a_1"), CSE6_i_i_i_ap2("i_2,i_1,i_3;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-2);
  I_i_i_i_ap2 = ArrayToT();  // release
  I_i_i_i_μ̃("i_1,i_2,i_3,μ̃_19915") = TA::einsum(g_i_i_Κ("i_3,i_2,Κ_1"), g_μ̃_i_Κ("μ̃_19915,i_1,Κ_1"), "i_1,i_2,i_3,μ̃_19915")("i_1,i_2,i_3,μ̃_19915");
  I_i_i_i_ap2("i_1,i_2,i_3;a_1") = TA::einsum(I_i_i_i_μ̃("i_1,i_2,i_3,μ̃_19915"), C_ap2_μ̃("i_1,i_2,μ̃_19915;a_1"), "i_1,i_2,i_3;a_1")("i_1,i_2,i_3;a_1");
  I_i_i_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_i_i_i_ap2("i_1,i_2,i_3;a_1"), CSE6_i_i_i_ap2("i_2,i_1,i_3;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-2);
  I_i_i_i_ap2 = ArrayToT();  // release
  CSE33_i_i_i_μ̃("i_3,i_4,i_2,μ̃_19772") = TA::einsum(g_i_i_Κ("i_3,i_2,Κ_1"), g_i_μ̃_Κ("i_4,μ̃_19772,Κ_1"), "i_3,i_4,i_2,μ̃_19772")("i_3,i_4,i_2,μ̃_19772");
  I_i_i("i_2,i_4") = TA::einsum(CSE33_i_i_i_μ̃("i_3,i_4,i_2,μ̃_19772"), CSE3_i_μ̃("i_3,μ̃_19772"), "i_2,i_4")("i_2,i_4");
  I2_i_i_ap2_ap2("i_1,i_2,i_4;a_2,a_5") = TA::einsum(I_i_i("i_2,i_4"), CSE4_i_i_i_ap2_ap2("i_4,i_2,i_1;a_2,a_5"), "i_1,i_2,i_4;a_2,a_5")("i_1,i_2,i_4;a_2,a_5");
  I_i_i = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I2_i_i_ap2_ap2("i_1,i_2,i_4;a_2,a_5"), CSE5_i_i_i_ap2_ap2("i_2,i_4,i_1;a_1,a_5"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (2);
  I2_i_i_ap2_ap2 = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(CSE8_i_i_i_ap2_ap2("i_1,i_3,i_2;a_2,a_4"), CSE5_i_i_i_ap2_ap2("i_2,i_3,i_1;a_1,a_4"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-2);
  I_i_i_μ̃_μ̃("i_3,i_4,μ̃_19743,μ̃_19744") = TA::einsum(g_i_μ̃_Κ("i_3,μ̃_19743,Κ_1"), g_i_μ̃_Κ("i_4,μ̃_19744,Κ_1"), "i_3,i_4,μ̃_19743,μ̃_19744")("i_3,i_4,μ̃_19743,μ̃_19744");
  CSE34_i_i_i_ap2_μ̃("i_2,i_3,i_4,μ̃_19744;a_3") = TA::einsum(I_i_i_μ̃_μ̃("i_3,i_4,μ̃_19743,μ̃_19744"), C_μ̃_ap2("i_2,i_4,μ̃_19743;a_3"), "i_2,i_3,i_4,μ̃_19744;a_3")("i_2,i_3,i_4,μ̃_19744;a_3");
  I_i_i_μ̃_μ̃ = TA::TSpArrayD();  // release
  I3_i_i_ap2_ap2("i_2,i_3,i_4;a_3,a_4") = TA::einsum(CSE34_i_i_i_ap2_μ̃("i_2,i_3,i_4,μ̃_19744;a_3"), C_μ̃_ap2("i_2,i_4,μ̃_19744;a_4"), "i_2,i_3,i_4;a_3,a_4")("i_2,i_3,i_4;a_3,a_4");
  I_i_i("i_2,i_3") = TA::einsum<TA::DeNest::True>(I3_i_i_ap2_ap2("i_2,i_3,i_4;a_3,a_4"), t_ap2_ap2_i_i("i_2,i_4;a_3,a_4"), "i_2,i_3")("i_2,i_3");
  I3_i_i_ap2_ap2 = ArrayToT();  // release
  I2_i_i_ap2_ap2("i_1,i_2,i_3;a_2,a_6") = TA::einsum(I_i_i("i_2,i_3"), CSE4_i_i_i_ap2_ap2("i_3,i_2,i_1;a_2,a_6"), "i_1,i_2,i_3;a_2,a_6")("i_1,i_2,i_3;a_2,a_6");
  I_i_i = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I2_i_i_ap2_ap2("i_1,i_2,i_3;a_2,a_6"), CSE5_i_i_i_ap2_ap2("i_2,i_3,i_1;a_1,a_6"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-4);
  I2_i_i_ap2_ap2 = ArrayToT();  // release
  CSE35_i_i_μ̃_μ̃("i_4,i_3,μ̃_19812,μ̃_19811") = TA::einsum(g_i_μ̃_Κ("i_3,μ̃_19811,Κ_1"), g_i_μ̃_Κ("i_4,μ̃_19812,Κ_1"), "i_4,i_3,μ̃_19812,μ̃_19811")("i_4,i_3,μ̃_19812,μ̃_19811");
  I_i_i_ap2_μ̃("i_2,i_3,i_4,μ̃_19812;a_3") = TA::einsum(CSE35_i_i_μ̃_μ̃("i_4,i_3,μ̃_19812,μ̃_19811"), C_μ̃_ap2("i_2,i_3,μ̃_19811;a_3"), "i_2,i_3,i_4,μ̃_19812;a_3")("i_2,i_3,i_4,μ̃_19812;a_3");
  I3_i_i_ap2_ap2("i_2,i_3,i_4;a_3,a_4") = TA::einsum(I_i_i_ap2_μ̃("i_2,i_3,i_4,μ̃_19812;a_3"), C_μ̃_ap2("i_2,i_3,μ̃_19812;a_4"), "i_2,i_3,i_4;a_3,a_4")("i_2,i_3,i_4;a_3,a_4");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_i_i("i_2,i_4") = TA::einsum<TA::DeNest::True>(I3_i_i_ap2_ap2("i_2,i_3,i_4;a_3,a_4"), t_ap2_ap2_i_i("i_2,i_3;a_3,a_4"), "i_2,i_4")("i_2,i_4");
  I3_i_i_ap2_ap2 = ArrayToT();  // release
  I2_i_i_ap2_ap2("i_1,i_2,i_4;a_2,a_6") = TA::einsum(I_i_i("i_2,i_4"), CSE4_i_i_i_ap2_ap2("i_4,i_2,i_1;a_2,a_6"), "i_1,i_2,i_4;a_2,a_6")("i_1,i_2,i_4;a_2,a_6");
  I_i_i = TA::TSpArrayD();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I2_i_i_ap2_ap2("i_1,i_2,i_4;a_2,a_6"), CSE5_i_i_i_ap2_ap2("i_2,i_4,i_1;a_1,a_6"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (2);
  I2_i_i_ap2_ap2 = ArrayToT();  // release
  I2_i_i_ap2_ap2("i_1,i_2,i_3;a_2,a_4") = TA::einsum(f_i_i("i_3,i_2"), CSE4_i_i_i_ap2_ap2("i_3,i_2,i_1;a_2,a_4"), "i_1,i_2,i_3;a_2,a_4")("i_1,i_2,i_3;a_2,a_4");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I2_i_i_ap2_ap2("i_1,i_2,i_3;a_2,a_4"), CSE5_i_i_i_ap2_ap2("i_2,i_3,i_1;a_1,a_4"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-2);
  I2_i_i_ap2_ap2 = ArrayToT();  // release
  I_i_i_μ̃_μ̃("i_1,i_3,μ̃_19735,μ̃_19736") = TA::einsum(g_i_i_Κ("i_3,i_1,Κ_1"), g_μ̃_μ̃_Κ("μ̃_19735,μ̃_19736,Κ_1"), "i_1,i_3,μ̃_19735,μ̃_19736")("i_1,i_3,μ̃_19735,μ̃_19736");
  I_i_i_ap2_μ̃("i_2,i_1,i_3,μ̃_19736;a_2") = TA::einsum(I_i_i_μ̃_μ̃("i_1,i_3,μ̃_19735,μ̃_19736"), C_ap2_μ̃("i_1,i_2,μ̃_19735;a_2"), "i_2,i_1,i_3,μ̃_19736;a_2")("i_2,i_1,i_3,μ̃_19736;a_2");
  I_i_i_μ̃_μ̃ = TA::TSpArrayD();  // release
  CSE36_i_i_i_ap2_ap2("i_2,i_3,i_1;a_2,a_3") = TA::einsum(I_i_i_ap2_μ̃("i_2,i_1,i_3,μ̃_19736;a_2"), C_μ̃_ap2("i_2,i_3,μ̃_19736;a_3"), "i_2,i_3,i_1;a_2,a_3")("i_2,i_3,i_1;a_2,a_3");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I2_i_i_ap2_ap2("i_3,i_1,i_2;a_2,a_4") = TA::einsum(CSE36_i_i_i_ap2_ap2("i_2,i_3,i_1;a_2,a_3"), t_ap2_ap2_i_i("i_2,i_3;a_3,a_4"), "i_3,i_1,i_2;a_2,a_4")("i_3,i_1,i_2;a_2,a_4");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I2_i_i_ap2_ap2("i_3,i_1,i_2;a_2,a_4"), CSE4_i_i_i_ap2_ap2("i_3,i_1,i_2;a_1,a_4"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-2);
  I2_i_i_ap2_ap2 = ArrayToT();  // release
  I2_i_i_ap2_ap2("i_3,i_1,i_2;a_1,a_4") = TA::einsum(CSE36_i_i_i_ap2_ap2("i_2,i_3,i_1;a_1,a_3"), t_ap2_ap2_i_i("i_3,i_2;a_3,a_4"), "i_3,i_1,i_2;a_1,a_4")("i_3,i_1,i_2;a_1,a_4");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I2_i_i_ap2_ap2("i_3,i_1,i_2;a_1,a_4"), CSE4_i_i_i_ap2_ap2("i_3,i_1,i_2;a_2,a_4"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-2);
  I2_i_i_ap2_ap2 = ArrayToT();  // release
  I_i_i_ap2_μ̃("i_1,i_2,μ̃_19705;a_1") = TA::einsum(CSE19_i_i_ap2_Κ("i_2,i_1,Κ_1;a_1"), CSE28_i_μ̃_Κ("i_2,μ̃_19705,Κ_1"), "i_1,i_2,μ̃_19705;a_1")("i_1,i_2,μ̃_19705;a_1");
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,μ̃_19705;a_1"), C_ap2_μ̃("i_1,i_2,μ̃_19705;a_2"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2");
  I_i_i_ap2_μ̃ = ArrayToT();  // release
  I_μ̃_μ̃("μ̃_19718,μ̃_19719") = TA::einsum(g_i_μ̃_Κ("i_3,μ̃_19718,Κ_1"), CSE28_i_μ̃_Κ("i_3,μ̃_19719,Κ_1"), "μ̃_19718,μ̃_19719")("μ̃_19718,μ̃_19719");
  I_ap2_μ̃("i_1,i_2,μ̃_19719;a_3") = TA::einsum(I_μ̃_μ̃("μ̃_19718,μ̃_19719"), C_μ̃_ap2("i_1,i_2,μ̃_19718;a_3"), "i_1,i_2,μ̃_19719;a_3")("i_1,i_2,μ̃_19719;a_3");
  I_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_ap2_ap2("i_1,i_2;a_2,a_3") = TA::einsum(I_ap2_μ̃("i_1,i_2,μ̃_19719;a_3"), C_ap2_μ̃("i_1,i_2,μ̃_19719;a_2"), "i_1,i_2;a_2,a_3")("i_1,i_2;a_2,a_3");
  I_ap2_μ̃ = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_ap2_ap2("i_1,i_2;a_2,a_3"), t_ap2_ap2_i_i("i_1,i_2;a_1,a_3"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (-2);
  I_ap2_ap2 = ArrayToT();  // release
  I_ap2_μ̃("i_1,i_2,μ̃_19818;a_2") = TA::einsum(f_μ̃_μ̃("μ̃_19817,μ̃_19818"), C_ap2_μ̃("i_1,i_2,μ̃_19817;a_2"), "i_1,i_2,μ̃_19818;a_2")("i_1,i_2,μ̃_19818;a_2");
  I_ap2_ap2("i_1,i_2;a_2,a_3") = TA::einsum(I_ap2_μ̃("i_1,i_2,μ̃_19818;a_2"), C_μ̃_ap2("i_1,i_2,μ̃_19818;a_3"), "i_1,i_2;a_2,a_3")("i_1,i_2;a_2,a_3");
  I_ap2_μ̃ = ArrayToT();  // release
  I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += (TA::einsum(I_ap2_ap2("i_1,i_2;a_2,a_3"), t_ap2_ap2_i_i("i_1,i_2;a_1,a_3"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2")) * (2);
  I_ap2_ap2 = ArrayToT();  // release
  // --- DF half-transform block (original lines 487-491), aux-Κ-batching gated ---
  {
    const char* _ats_env = std::getenv("SPTC_AUX_TARGET_SIZE");
    const std::size_t _ats = _ats_env ? std::strtoul(_ats_env, nullptr, 10) : 0;
    if (_ats > 0) {
      // Stream Κ in batches; never forms the giant μ̃Κ intermediate whole.
      accumulate_df_halftransform_batched(I_i_i_ap2_ap2, g_μ̃_μ̃_Κ, C_ap2_μ̃,
                                          C_μ̃_ap2, t_ap2_ap2_i_i, _ats);
    } else {
      // Verbatim original (bit-identical to the shipped generated residual).
      I_ap2_μ̃_Κ("i_1,i_2,μ̃_19906,Κ_1;a_1") = TA::einsum(g_μ̃_μ̃_Κ("μ̃_19905,μ̃_19906,Κ_1"), C_ap2_μ̃("i_1,i_2,μ̃_19905;a_1"), "i_1,i_2,μ̃_19906,Κ_1;a_1")("i_1,i_2,μ̃_19906,Κ_1;a_1");
      CSE37_i_i_ap2_ap2_Κ("i_2,i_1,Κ_1;a_1,a_3") = TA::einsum(I_ap2_μ̃_Κ("i_1,i_2,μ̃_19906,Κ_1;a_1"), C_μ̃_ap2("i_1,i_2,μ̃_19906;a_3"), "i_2,i_1,Κ_1;a_1,a_3")("i_2,i_1,Κ_1;a_1,a_3");
      I_ap2_μ̃_Κ = ArrayToT();  // release
      I_i_i_ap2_ap2_Κ("i_1,i_2,Κ_1;a_1,a_4") = TA::einsum(CSE37_i_i_ap2_ap2_Κ("i_2,i_1,Κ_1;a_1,a_3"), t_ap2_ap2_i_i("i_1,i_2;a_3,a_4"), "i_1,i_2,Κ_1;a_1,a_4")("i_1,i_2,Κ_1;a_1,a_4");
      I_i_i_ap2_ap2("i_1,i_2;a_1,a_2") += TA::einsum(I_i_i_ap2_ap2_Κ("i_1,i_2,Κ_1;a_1,a_4"), CSE37_i_i_ap2_ap2_Κ("i_2,i_1,Κ_1;a_2,a_4"), "i_1,i_2;a_1,a_2")("i_1,i_2;a_1,a_2");
      I_i_i_ap2_ap2_Κ = ArrayToT();  // release
    }
  }
  return I_i_i_ap2_ap2;
}
