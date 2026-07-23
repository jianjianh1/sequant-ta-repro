// AUTO-GENERATED (2026-07-19) by SeQuant's native TiledArrayGenerator
// (sptc-bench/scratch/sequant-patched/SeQuant/core/export/
// tiledarray_generator.hpp) from the full cck.ipp-matching closed-shell
// CSV-CCSD T1 residual derivation (tests/manual/test_csv_ccsd_derivation.cpp,
// 26 summands). See ta_sequant_native_residual.h for the leaf-parameter ->
// TATensors field mapping. Regenerate + re-paste (do not hand-edit) if the
// derivation pipeline changes.
#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>
#include <cmath>

TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy> whole_t1_residual(const TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>& C_μ̃_ap1, const TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>& t_ap1_i, const TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>& C_ap1_μ̃, const TA::TSpArrayD& f_i_i, const TA::TSpArrayD& s_μ̃_μ̃, const TA::TSpArrayD& g_i_μ̃_Κ, const TA::TSpArrayD& g_μ̃_i_Κ, const TA::TSpArrayD& g_μ̃_μ̃_Κ, const TA::TSpArrayD& g_i_i_Κ, const TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>& C_μ̃_ap2, const TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>& t_ap2_ap2_i_i, const TA::TSpArrayD& f_i_μ̃, const TA::TSpArrayD& f_μ̃_μ̃, const TA::TSpArrayD& f_μ̃_i) {
  TA::TSpArrayD CSE1_i_μ̃;
  TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy> I_i_ap1;
  TA::TSpArrayD I_i_μ̃;
  TA::TSpArrayD I2_i_μ̃;
  TA::TSpArrayD CSE2_Κ;
  TA::TSpArrayD I_μ̃_μ̃;
  TA::TSpArrayD CSE3_i_μ̃;
  TA::TSpArrayD I_i_i;
  TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy> CSE4_i_i_ap2;
  TA::TSpArrayD I_Κ;
  TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy> I2_i_ap2;
  TA::TSpArrayD CSE5_i_μ̃;
  TA::TSpArrayD CSE6_i_i_Κ;
  TA::TSpArrayD I_i_μ̃_Κ;
  TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy> CSE7_i_i_i_ap2;
  TA::TSpArrayD I_i_i_Κ;
  TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy> I_i_ap2_Κ;
  TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy> CSE8_i_i_ap2_Κ;
  TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy> CSE9_i_i_ap2_μ̃;
  TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy> CSE10_i_i_ap2;
  TA::TSpArrayD CSE11_i_i_μ̃_μ̃;
  TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy> I_i_i_ap2_ap2;
  TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy> I_i_i_ap2_μ̃;
  TA::TSpArrayD CSE12_i_i_i_μ̃;
  TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy> I2_i_ap1;
  TA::TSpArrayD I_i_i_μ̃_μ̃;
  TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy> CSE13_i_i_i_ap2;
  TA::TSpArrayD I_i_i_i_μ̃;
  TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy> CSE14_i_i_ap2;
  CSE1_i_μ̃("i_2,μ̃_19580") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19580;a_2"), t_ap1_i("i_2;a_2"), "i_2,μ̃_19580")("i_2,μ̃_19580");
  I2_i_μ̃("i_1,μ̃_19580") = TA::einsum(f_i_i("i_2,i_1"), CSE1_i_μ̃("i_2,μ̃_19580"), "i_1,μ̃_19580")("i_1,μ̃_19580");
  I_i_μ̃("i_1,μ̃_19579") = TA::einsum(I2_i_μ̃("i_1,μ̃_19580"), s_μ̃_μ̃("μ̃_19579,μ̃_19580"), "i_1,μ̃_19579")("i_1,μ̃_19579");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") = (TA::einsum(I_i_μ̃("i_1,μ̃_19579"), C_ap1_μ̃("i_1,μ̃_19579;a_1"), "i_1;a_1")("i_1;a_1")) * (-1);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_2,μ̃_19581") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19581;a_2"), t_ap1_i("i_2;a_2"), "i_2,μ̃_19581")("i_2,μ̃_19581");
  CSE2_Κ("Κ_1") = TA::einsum(I_i_μ̃("i_2,μ̃_19581"), g_i_μ̃_Κ("i_2,μ̃_19581,Κ_1"), "Κ_1")("Κ_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_1,μ̃_19582") = TA::einsum(CSE2_Κ("Κ_1"), g_μ̃_i_Κ("μ̃_19582,i_1,Κ_1"), "i_1,μ̃_19582")("i_1,μ̃_19582");
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19582"), C_ap1_μ̃("i_1,μ̃_19582;a_1"), "i_1;a_1")("i_1;a_1")) * (2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_μ̃_μ̃("μ̃_19587,μ̃_19588") = TA::einsum(CSE2_Κ("Κ_1"), g_μ̃_μ̃_Κ("μ̃_19587,μ̃_19588,Κ_1"), "μ̃_19587,μ̃_19588")("μ̃_19587,μ̃_19588");
  I_i_μ̃("i_1,μ̃_19587") = TA::einsum(I_μ̃_μ̃("μ̃_19587,μ̃_19588"), CSE1_i_μ̃("i_1,μ̃_19588"), "i_1,μ̃_19587")("i_1,μ̃_19587");
  I_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19587"), C_ap1_μ̃("i_1,μ̃_19587;a_1"), "i_1;a_1")("i_1;a_1")) * (2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_2,μ̃_19625") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19625;a_3"), t_ap1_i("i_2;a_3"), "i_2,μ̃_19625")("i_2,μ̃_19625");
  CSE3_i_μ̃("i_2,μ̃_19624") = TA::einsum(I_i_μ̃("i_2,μ̃_19625"), s_μ̃_μ̃("μ̃_19624,μ̃_19625"), "i_2,μ̃_19624")("i_2,μ̃_19624");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i("i_1,i_2") = TA::einsum(g_i_i_Κ("i_2,i_1,Κ_1"), CSE2_Κ("Κ_1"), "i_1,i_2")("i_1,i_2");
  I_i_μ̃("i_1,μ̃_19624") = TA::einsum(I_i_i("i_1,i_2"), CSE3_i_μ̃("i_2,μ̃_19624"), "i_1,μ̃_19624")("i_1,μ̃_19624");
  I_i_i = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19624"), C_ap1_μ̃("i_1,μ̃_19624;a_1"), "i_1;a_1")("i_1;a_1")) * (-2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_μ̃("i_2,μ̃_19626") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19626;a_2"), t_ap1_i("i_2;a_2"), "i_2,μ̃_19626")("i_2,μ̃_19626");
  I_Κ("Κ_1") = TA::einsum(I2_i_μ̃("i_2,μ̃_19626"), g_i_μ̃_Κ("i_2,μ̃_19626,Κ_1"), "Κ_1")("Κ_1");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_3,μ̃_19627") = TA::einsum(I_Κ("Κ_1"), g_i_μ̃_Κ("i_3,μ̃_19627,Κ_1"), "i_3,μ̃_19627")("i_3,μ̃_19627");
  I_Κ = TA::TSpArrayD();  // release
  CSE4_i_i_ap2("i_1,i_3;a_3") = TA::einsum(I_i_μ̃("i_3,μ̃_19627"), C_μ̃_ap2("i_1,i_3,μ̃_19627;a_3"), "i_1,i_3;a_3")("i_1,i_3;a_3");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_ap2("i_3,i_1;a_4") = TA::einsum(CSE4_i_i_ap2("i_1,i_3;a_3"), t_ap2_ap2_i_i("i_3,i_1;a_3,a_4"), "i_3,i_1;a_4")("i_3,i_1;a_4");
  I2_i_μ̃("i_1,μ̃_19629") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_3,i_1;a_4"), C_μ̃_ap2("i_1,i_3,μ̃_19629;a_4"), "i_1,μ̃_19629")("i_1,μ̃_19629");
  I2_i_ap2 = TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>();  // release
  I_i_μ̃("i_1,μ̃_19628") = TA::einsum(I2_i_μ̃("i_1,μ̃_19629"), s_μ̃_μ̃("μ̃_19628,μ̃_19629"), "i_1,μ̃_19628")("i_1,μ̃_19628");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19628"), C_ap1_μ̃("i_1,μ̃_19628;a_1"), "i_1;a_1")("i_1;a_1")) * (4);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_2,μ̃_19642") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_2,μ̃_19642;a_2"), t_ap1_i("i_2;a_2"), "i_2,μ̃_19642")("i_2,μ̃_19642");
  I_Κ("Κ_1") = TA::einsum(I_i_μ̃("i_2,μ̃_19642"), g_i_μ̃_Κ("i_2,μ̃_19642,Κ_1"), "Κ_1")("Κ_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  CSE5_i_μ̃("i_3,μ̃_19643") = TA::einsum(I_Κ("Κ_1"), g_i_μ̃_Κ("i_3,μ̃_19643,Κ_1"), "i_3,μ̃_19643")("i_3,μ̃_19643");
  I_Κ = TA::TSpArrayD();  // release
  I_i_i("i_1,i_3") = TA::einsum(CSE5_i_μ̃("i_3,μ̃_19643"), CSE1_i_μ̃("i_1,μ̃_19643"), "i_1,i_3")("i_1,i_3");
  I_i_μ̃("i_1,μ̃_19644") = TA::einsum(I_i_i("i_1,i_3"), CSE3_i_μ̃("i_3,μ̃_19644"), "i_1,μ̃_19644")("i_1,μ̃_19644");
  I_i_i = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19644"), C_ap1_μ̃("i_1,μ̃_19644;a_1"), "i_1;a_1")("i_1;a_1")) * (-2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_ap2("i_3,i_1;a_4") = TA::einsum(CSE4_i_i_ap2("i_1,i_3;a_3"), t_ap2_ap2_i_i("i_1,i_3;a_3,a_4"), "i_3,i_1;a_4")("i_3,i_1;a_4");
  I2_i_μ̃("i_1,μ̃_19656") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_3,i_1;a_4"), C_μ̃_ap2("i_1,i_3,μ̃_19656;a_4"), "i_1,μ̃_19656")("i_1,μ̃_19656");
  I2_i_ap2 = TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>();  // release
  I_i_μ̃("i_1,μ̃_19655") = TA::einsum(I2_i_μ̃("i_1,μ̃_19656"), s_μ̃_μ̃("μ̃_19655,μ̃_19656"), "i_1,μ̃_19655")("i_1,μ̃_19655");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19655"), C_ap1_μ̃("i_1,μ̃_19655;a_1"), "i_1;a_1")("i_1;a_1")) * (-2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_1,μ̃_19583") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_1,μ̃_19583;a_2"), t_ap1_i("i_1;a_2"), "i_1,μ̃_19583")("i_1,μ̃_19583");
  CSE6_i_i_Κ("i_2,i_1,Κ_1") = TA::einsum(I_i_μ̃("i_1,μ̃_19583"), g_i_μ̃_Κ("i_2,μ̃_19583,Κ_1"), "i_2,i_1,Κ_1")("i_2,i_1,Κ_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃_Κ("i_2,μ̃_19584,Κ_1") = TA::einsum(g_μ̃_μ̃_Κ("μ̃_19584,μ̃_19585,Κ_1"), CSE1_i_μ̃("i_2,μ̃_19585"), "i_2,μ̃_19584,Κ_1")("i_2,μ̃_19584,Κ_1");
  I_i_μ̃("i_1,μ̃_19584") = TA::einsum(I_i_μ̃_Κ("i_2,μ̃_19584,Κ_1"), CSE6_i_i_Κ("i_2,i_1,Κ_1"), "i_1,μ̃_19584")("i_1,μ̃_19584");
  I_i_μ̃_Κ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19584"), C_ap1_μ̃("i_1,μ̃_19584;a_1"), "i_1;a_1")("i_1;a_1")) * (-1);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_1,μ̃_19592") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_1,μ̃_19592;a_2"), t_ap1_i("i_1;a_2"), "i_1,μ̃_19592")("i_1,μ̃_19592");
  I_i_i_Κ("i_1,i_2,Κ_1") = TA::einsum(I_i_μ̃("i_1,μ̃_19592"), g_i_μ̃_Κ("i_2,μ̃_19592,Κ_1"), "i_1,i_2,Κ_1")("i_1,i_2,Κ_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap2_Κ("i_2,i_3,Κ_1;a_3") = TA::einsum(g_i_μ̃_Κ("i_3,μ̃_19593,Κ_1"), C_μ̃_ap2("i_2,i_3,μ̃_19593;a_3"), "i_2,i_3,Κ_1;a_3")("i_2,i_3,Κ_1;a_3");
  CSE7_i_i_i_ap2("i_3,i_2,i_1;a_3") = TA::einsum(I_i_i_Κ("i_1,i_2,Κ_1"), I_i_ap2_Κ("i_2,i_3,Κ_1;a_3"), "i_3,i_2,i_1;a_3")("i_3,i_2,i_1;a_3");
  I_i_ap2_Κ = TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>();  // release
  I_i_i_Κ = TA::TSpArrayD();  // release
  I2_i_ap2("i_2,i_3,i_1;a_4") = TA::einsum(CSE7_i_i_i_ap2("i_3,i_2,i_1;a_3"), t_ap2_ap2_i_i("i_3,i_2;a_3,a_4"), "i_2,i_3,i_1;a_4")("i_2,i_3,i_1;a_4");
  I2_i_μ̃("i_1,μ̃_19595") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_2,i_3,i_1;a_4"), C_μ̃_ap2("i_2,i_3,μ̃_19595;a_4"), "i_1,μ̃_19595")("i_1,μ̃_19595");
  I2_i_ap2 = TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>();  // release
  I_i_μ̃("i_1,μ̃_19594") = TA::einsum(I2_i_μ̃("i_1,μ̃_19595"), s_μ̃_μ̃("μ̃_19594,μ̃_19595"), "i_1,μ̃_19594")("i_1,μ̃_19594");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19594"), C_ap1_μ̃("i_1,μ̃_19594;a_1"), "i_1;a_1")("i_1;a_1")) * (-2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  CSE8_i_i_ap2_Κ("i_1,i_2,Κ_1;a_2") = TA::einsum(g_i_μ̃_Κ("i_2,μ̃_19614,Κ_1"), C_μ̃_ap2("i_1,i_2,μ̃_19614;a_2"), "i_1,i_2,Κ_1;a_2")("i_1,i_2,Κ_1;a_2");
  CSE9_i_i_ap2_μ̃("i_1,i_2,μ̃_19616;a_2") = TA::einsum(C_μ̃_ap2("i_1,i_2,μ̃_19616;a_3"), t_ap2_ap2_i_i("i_2,i_1;a_2,a_3"), "i_1,i_2,μ̃_19616;a_2")("i_1,i_2,μ̃_19616;a_2");
  I_i_μ̃_Κ("i_1,μ̃_19616,Κ_1") = TA::einsum<TA::DeNest::True>(CSE8_i_i_ap2_Κ("i_1,i_2,Κ_1;a_2"), CSE9_i_i_ap2_μ̃("i_1,i_2,μ̃_19616;a_2"), "i_1,μ̃_19616,Κ_1")("i_1,μ̃_19616,Κ_1");
  I_i_μ̃("i_1,μ̃_19615") = TA::einsum(I_i_μ̃_Κ("i_1,μ̃_19616,Κ_1"), g_μ̃_μ̃_Κ("μ̃_19615,μ̃_19616,Κ_1"), "i_1,μ̃_19615")("i_1,μ̃_19615");
  I_i_μ̃_Κ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19615"), C_ap1_μ̃("i_1,μ̃_19615;a_1"), "i_1;a_1")("i_1;a_1")) * (2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃_Κ("i_1,μ̃_19648,Κ_1") = TA::einsum<TA::DeNest::True>(CSE8_i_i_ap2_Κ("i_1,i_2,Κ_1;a_2"), CSE9_i_i_ap2_μ̃("i_2,i_1,μ̃_19648;a_2"), "i_1,μ̃_19648,Κ_1")("i_1,μ̃_19648,Κ_1");
  I_i_μ̃("i_1,μ̃_19647") = TA::einsum(I_i_μ̃_Κ("i_1,μ̃_19648,Κ_1"), g_μ̃_μ̃_Κ("μ̃_19647,μ̃_19648,Κ_1"), "i_1,μ̃_19647")("i_1,μ̃_19647");
  I_i_μ̃_Κ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19647"), C_ap1_μ̃("i_1,μ̃_19647;a_1"), "i_1;a_1")("i_1;a_1")) * (-1);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_ap2("i_2,i_3,i_1;a_4") = TA::einsum(CSE7_i_i_i_ap2("i_3,i_2,i_1;a_3"), t_ap2_ap2_i_i("i_2,i_3;a_3,a_4"), "i_2,i_3,i_1;a_4")("i_2,i_3,i_1;a_4");
  I2_i_μ̃("i_1,μ̃_19641") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_2,i_3,i_1;a_4"), C_μ̃_ap2("i_2,i_3,μ̃_19641;a_4"), "i_1,μ̃_19641")("i_1,μ̃_19641");
  I2_i_ap2 = TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>();  // release
  I_i_μ̃("i_1,μ̃_19640") = TA::einsum(I2_i_μ̃("i_1,μ̃_19641"), s_μ̃_μ̃("μ̃_19640,μ̃_19641"), "i_1,μ̃_19640")("i_1,μ̃_19640");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += TA::einsum(I_i_μ̃("i_1,μ̃_19640"), C_ap1_μ̃("i_1,μ̃_19640;a_1"), "i_1;a_1")("i_1;a_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_μ̃("i_3,μ̃_19619") = TA::einsum(g_i_μ̃_Κ("i_2,μ̃_19619,Κ_1"), CSE6_i_i_Κ("i_3,i_2,Κ_1"), "i_3,μ̃_19619")("i_3,μ̃_19619");
  I_i_i("i_1,i_3") = TA::einsum(I2_i_μ̃("i_3,μ̃_19619"), CSE1_i_μ̃("i_1,μ̃_19619"), "i_1,i_3")("i_1,i_3");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_1,μ̃_19621") = TA::einsum(I_i_i("i_1,i_3"), CSE3_i_μ̃("i_3,μ̃_19621"), "i_1,μ̃_19621")("i_1,μ̃_19621");
  I_i_i = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += TA::einsum(I_i_μ̃("i_1,μ̃_19621"), C_ap1_μ̃("i_1,μ̃_19621;a_1"), "i_1;a_1")("i_1;a_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_μ̃("i_3,μ̃_19634") = TA::einsum<TA::DeNest::True>(C_μ̃_ap1("i_3,μ̃_19634;a_2"), t_ap1_i("i_3;a_2"), "i_3,μ̃_19634")("i_3,μ̃_19634");
  I_i_i_Κ("i_2,i_3,Κ_1") = TA::einsum(I2_i_μ̃("i_3,μ̃_19634"), g_i_μ̃_Κ("i_2,μ̃_19634,Κ_1"), "i_2,i_3,Κ_1")("i_2,i_3,Κ_1");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_2,μ̃_19635") = TA::einsum(I_i_i_Κ("i_2,i_3,Κ_1"), g_i_μ̃_Κ("i_3,μ̃_19635,Κ_1"), "i_2,μ̃_19635")("i_2,μ̃_19635");
  I_i_i_Κ = TA::TSpArrayD();  // release
  CSE10_i_i_ap2("i_1,i_2;a_3") = TA::einsum(I_i_μ̃("i_2,μ̃_19635"), C_μ̃_ap2("i_1,i_2,μ̃_19635;a_3"), "i_1,i_2;a_3")("i_1,i_2;a_3");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_ap2("i_2,i_1;a_4") = TA::einsum(CSE10_i_i_ap2("i_1,i_2;a_3"), t_ap2_ap2_i_i("i_2,i_1;a_3,a_4"), "i_2,i_1;a_4")("i_2,i_1;a_4");
  I2_i_μ̃("i_1,μ̃_19637") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_2,i_1;a_4"), C_μ̃_ap2("i_1,i_2,μ̃_19637;a_4"), "i_1,μ̃_19637")("i_1,μ̃_19637");
  I2_i_ap2 = TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>();  // release
  I_i_μ̃("i_1,μ̃_19636") = TA::einsum(I2_i_μ̃("i_1,μ̃_19637"), s_μ̃_μ̃("μ̃_19636,μ̃_19637"), "i_1,μ̃_19636")("i_1,μ̃_19636");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19636"), C_ap1_μ̃("i_1,μ̃_19636;a_1"), "i_1;a_1")("i_1;a_1")) * (-2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_ap2("i_2,i_1;a_4") = TA::einsum(CSE10_i_i_ap2("i_1,i_2;a_3"), t_ap2_ap2_i_i("i_1,i_2;a_3,a_4"), "i_2,i_1;a_4")("i_2,i_1;a_4");
  I2_i_μ̃("i_1,μ̃_19660") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_2,i_1;a_4"), C_μ̃_ap2("i_1,i_2,μ̃_19660;a_4"), "i_1,μ̃_19660")("i_1,μ̃_19660");
  I2_i_ap2 = TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>();  // release
  I_i_μ̃("i_1,μ̃_19659") = TA::einsum(I2_i_μ̃("i_1,μ̃_19660"), s_μ̃_μ̃("μ̃_19659,μ̃_19660"), "i_1,μ̃_19659")("i_1,μ̃_19659");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += TA::einsum(I_i_μ̃("i_1,μ̃_19659"), C_ap1_μ̃("i_1,μ̃_19659;a_1"), "i_1;a_1")("i_1;a_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  CSE11_i_i_μ̃_μ̃("i_3,i_2,μ̃_19597,μ̃_19596") = TA::einsum(g_i_μ̃_Κ("i_2,μ̃_19596,Κ_1"), g_i_μ̃_Κ("i_3,μ̃_19597,Κ_1"), "i_3,i_2,μ̃_19597,μ̃_19596")("i_3,i_2,μ̃_19597,μ̃_19596");
  I_i_i_ap2_μ̃("i_1,i_2,i_3,μ̃_19597;a_2") = TA::einsum(CSE11_i_i_μ̃_μ̃("i_3,i_2,μ̃_19597,μ̃_19596"), C_μ̃_ap2("i_1,i_2,μ̃_19596;a_2"), "i_1,i_2,i_3,μ̃_19597;a_2")("i_1,i_2,i_3,μ̃_19597;a_2");
  I_i_i_ap2_ap2("i_1,i_2,i_3;a_2,a_3") = TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,i_3,μ̃_19597;a_2"), C_μ̃_ap2("i_1,i_2,μ̃_19597;a_3"), "i_1,i_2,i_3;a_2,a_3")("i_1,i_2,i_3;a_2,a_3");
  I_i_i_ap2_μ̃ = TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>();  // release
  I_i_i("i_1,i_3") = TA::einsum<TA::DeNest::True>(I_i_i_ap2_ap2("i_1,i_2,i_3;a_2,a_3"), t_ap2_ap2_i_i("i_1,i_2;a_2,a_3"), "i_1,i_3")("i_1,i_3");
  I_i_i_ap2_ap2 = TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>();  // release
  I_i_μ̃("i_1,μ̃_19598") = TA::einsum(I_i_i("i_1,i_3"), CSE3_i_μ̃("i_3,μ̃_19598"), "i_1,μ̃_19598")("i_1,μ̃_19598");
  I_i_i = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += TA::einsum(I_i_μ̃("i_1,μ̃_19598"), C_ap1_μ̃("i_1,μ̃_19598;a_1"), "i_1;a_1")("i_1;a_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_ap2_μ̃("i_1,i_2,i_3,μ̃_19650;a_2") = TA::einsum(CSE11_i_i_μ̃_μ̃("i_3,i_2,μ̃_19650,μ̃_19649"), C_μ̃_ap2("i_1,i_3,μ̃_19649;a_2"), "i_1,i_2,i_3,μ̃_19650;a_2")("i_1,i_2,i_3,μ̃_19650;a_2");
  I_i_i_ap2_ap2("i_1,i_2,i_3;a_2,a_3") = TA::einsum(I_i_i_ap2_μ̃("i_1,i_2,i_3,μ̃_19650;a_2"), C_μ̃_ap2("i_1,i_3,μ̃_19650;a_3"), "i_1,i_2,i_3;a_2,a_3")("i_1,i_2,i_3;a_2,a_3");
  I_i_i_ap2_μ̃ = TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>();  // release
  I_i_i("i_1,i_2") = TA::einsum<TA::DeNest::True>(I_i_i_ap2_ap2("i_1,i_2,i_3;a_2,a_3"), t_ap2_ap2_i_i("i_1,i_3;a_2,a_3"), "i_1,i_2")("i_1,i_2");
  I_i_i_ap2_ap2 = TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>();  // release
  I_i_μ̃("i_1,μ̃_19651") = TA::einsum(I_i_i("i_1,i_2"), CSE3_i_μ̃("i_2,μ̃_19651"), "i_1,μ̃_19651")("i_1,μ̃_19651");
  I_i_i = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19651"), C_ap1_μ̃("i_1,μ̃_19651;a_1"), "i_1;a_1")("i_1;a_1")) * (-2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  CSE12_i_i_i_μ̃("i_2,i_3,i_1,μ̃_19608") = TA::einsum(g_i_i_Κ("i_2,i_1,Κ_1"), g_i_μ̃_Κ("i_3,μ̃_19608,Κ_1"), "i_2,i_3,i_1,μ̃_19608")("i_2,i_3,i_1,μ̃_19608");
  I_i_i("i_1,i_3") = TA::einsum(CSE12_i_i_i_μ̃("i_2,i_3,i_1,μ̃_19608"), CSE1_i_μ̃("i_2,μ̃_19608"), "i_1,i_3")("i_1,i_3");
  I_i_μ̃("i_1,μ̃_19609") = TA::einsum(I_i_i("i_1,i_3"), CSE3_i_μ̃("i_3,μ̃_19609"), "i_1,μ̃_19609")("i_1,μ̃_19609");
  I_i_i = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += TA::einsum(I_i_μ̃("i_1,μ̃_19609"), C_ap1_μ̃("i_1,μ̃_19609;a_1"), "i_1;a_1")("i_1;a_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_ap1("i_1,i_2;a_2") = TA::einsum(f_i_μ̃("i_2,μ̃_19630"), C_μ̃_ap1("i_1,μ̃_19630;a_2"), "i_1,i_2;a_2")("i_1,i_2;a_2");
  I_i_i("i_1,i_2") = TA::einsum<TA::DeNest::True>(I2_i_ap1("i_1,i_2;a_2"), t_ap1_i("i_1;a_2"), "i_1,i_2")("i_1,i_2");
  I2_i_ap1 = TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>();  // release
  I_i_μ̃("i_1,μ̃_19631") = TA::einsum(I_i_i("i_1,i_2"), CSE3_i_μ̃("i_2,μ̃_19631"), "i_1,μ̃_19631")("i_1,μ̃_19631");
  I_i_i = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19631"), C_ap1_μ̃("i_1,μ̃_19631;a_1"), "i_1;a_1")("i_1;a_1")) * (-1);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_μ̃_μ̃("i_1,i_2,μ̃_19606,μ̃_19607") = TA::einsum(g_i_i_Κ("i_2,i_1,Κ_1"), g_μ̃_μ̃_Κ("μ̃_19606,μ̃_19607,Κ_1"), "i_1,i_2,μ̃_19606,μ̃_19607")("i_1,i_2,μ̃_19606,μ̃_19607");
  I_i_μ̃("i_1,μ̃_19606") = TA::einsum(I_i_i_μ̃_μ̃("i_1,i_2,μ̃_19606,μ̃_19607"), CSE1_i_μ̃("i_2,μ̃_19607"), "i_1,μ̃_19606")("i_1,μ̃_19606");
  I_i_i_μ̃_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19606"), C_ap1_μ̃("i_1,μ̃_19606;a_1"), "i_1;a_1")("i_1;a_1")) * (-1);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_μ̃("i_1,μ̃_19617") = TA::einsum(f_μ̃_μ̃("μ̃_19617,μ̃_19618"), CSE1_i_μ̃("i_1,μ̃_19618"), "i_1,μ̃_19617")("i_1,μ̃_19617");
  I_i_ap1("i_1;a_1") += TA::einsum(I_i_μ̃("i_1,μ̃_19617"), C_ap1_μ̃("i_1,μ̃_19617;a_1"), "i_1;a_1")("i_1;a_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_i_i_μ̃("i_1,i_2,i_3,μ̃_19589") = TA::einsum(g_i_i_Κ("i_2,i_1,Κ_1"), g_i_μ̃_Κ("i_3,μ̃_19589,Κ_1"), "i_1,i_2,i_3,μ̃_19589")("i_1,i_2,i_3,μ̃_19589");
  CSE13_i_i_i_ap2("i_3,i_2,i_1;a_2") = TA::einsum(I_i_i_i_μ̃("i_1,i_2,i_3,μ̃_19589"), C_μ̃_ap2("i_2,i_3,μ̃_19589;a_2"), "i_3,i_2,i_1;a_2")("i_3,i_2,i_1;a_2");
  I_i_i_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_ap2("i_2,i_3,i_1;a_3") = TA::einsum(CSE13_i_i_i_ap2("i_3,i_2,i_1;a_2"), t_ap2_ap2_i_i("i_3,i_2;a_2,a_3"), "i_2,i_3,i_1;a_3")("i_2,i_3,i_1;a_3");
  I2_i_μ̃("i_1,μ̃_19591") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_2,i_3,i_1;a_3"), C_μ̃_ap2("i_2,i_3,μ̃_19591;a_3"), "i_1,μ̃_19591")("i_1,μ̃_19591");
  I2_i_ap2 = TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>();  // release
  I_i_μ̃("i_1,μ̃_19590") = TA::einsum(I2_i_μ̃("i_1,μ̃_19591"), s_μ̃_μ̃("μ̃_19590,μ̃_19591"), "i_1,μ̃_19590")("i_1,μ̃_19590");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19590"), C_ap1_μ̃("i_1,μ̃_19590;a_1"), "i_1;a_1")("i_1;a_1")) * (-2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_ap2("i_2,i_3,i_1;a_3") = TA::einsum(CSE13_i_i_i_ap2("i_3,i_2,i_1;a_2"), t_ap2_ap2_i_i("i_2,i_3;a_2,a_3"), "i_2,i_3,i_1;a_3")("i_2,i_3,i_1;a_3");
  I2_i_μ̃("i_1,μ̃_19613") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_2,i_3,i_1;a_3"), C_μ̃_ap2("i_2,i_3,μ̃_19613;a_3"), "i_1,μ̃_19613")("i_1,μ̃_19613");
  I2_i_ap2 = TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>();  // release
  I_i_μ̃("i_1,μ̃_19612") = TA::einsum(I2_i_μ̃("i_1,μ̃_19613"), s_μ̃_μ̃("μ̃_19612,μ̃_19613"), "i_1,μ̃_19612")("i_1,μ̃_19612");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += TA::einsum(I_i_μ̃("i_1,μ̃_19612"), C_ap1_μ̃("i_1,μ̃_19612;a_1"), "i_1;a_1")("i_1;a_1");
  I_i_μ̃ = TA::TSpArrayD();  // release
  CSE14_i_i_ap2("i_1,i_2;a_2") = TA::einsum(f_i_μ̃("i_2,μ̃_19600"), C_μ̃_ap2("i_1,i_2,μ̃_19600;a_2"), "i_1,i_2;a_2")("i_1,i_2;a_2");
  I2_i_ap2("i_2,i_1;a_3") = TA::einsum(CSE14_i_i_ap2("i_1,i_2;a_2"), t_ap2_ap2_i_i("i_1,i_2;a_2,a_3"), "i_2,i_1;a_3")("i_2,i_1;a_3");
  I2_i_μ̃("i_1,μ̃_19602") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_2,i_1;a_3"), C_μ̃_ap2("i_1,i_2,μ̃_19602;a_3"), "i_1,μ̃_19602")("i_1,μ̃_19602");
  I2_i_ap2 = TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>();  // release
  I_i_μ̃("i_1,μ̃_19601") = TA::einsum(I2_i_μ̃("i_1,μ̃_19602"), s_μ̃_μ̃("μ̃_19601,μ̃_19602"), "i_1,μ̃_19601")("i_1,μ̃_19601");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19601"), C_ap1_μ̃("i_1,μ̃_19601;a_1"), "i_1;a_1")("i_1;a_1")) * (-1);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I2_i_ap2("i_2,i_1;a_3") = TA::einsum(CSE14_i_i_ap2("i_1,i_2;a_2"), t_ap2_ap2_i_i("i_2,i_1;a_2,a_3"), "i_2,i_1;a_3")("i_2,i_1;a_3");
  I2_i_μ̃("i_1,μ̃_19605") = TA::einsum<TA::DeNest::True>(I2_i_ap2("i_2,i_1;a_3"), C_μ̃_ap2("i_1,i_2,μ̃_19605;a_3"), "i_1,μ̃_19605")("i_1,μ̃_19605");
  I2_i_ap2 = TA::DistArray<TA::Tensor<TA::ArenaTensor<double>>, TA::SparsePolicy>();  // release
  I_i_μ̃("i_1,μ̃_19604") = TA::einsum(I2_i_μ̃("i_1,μ̃_19605"), s_μ̃_μ̃("μ̃_19604,μ̃_19605"), "i_1,μ̃_19604")("i_1,μ̃_19604");
  I2_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += (TA::einsum(I_i_μ̃("i_1,μ̃_19604"), C_ap1_μ̃("i_1,μ̃_19604;a_1"), "i_1;a_1")("i_1;a_1")) * (2);
  I_i_μ̃ = TA::TSpArrayD();  // release
  I_i_ap1("i_1;a_1") += TA::einsum(f_μ̃_i("μ̃_19633,i_1"), C_ap1_μ̃("i_1,μ̃_19633;a_1"), "i_1;a_1")("i_1;a_1");
  return I_i_ap1;
}

