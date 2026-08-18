#ifndef SPTC_TA_TENSORS_H
#define SPTC_TA_TENSORS_H

#include <tiledarray.h>

// Tensor-of-tensor type for the four PNO/OSV-restricted leaves (C1, C2, T1,
// T2) — outer indices are block-sparse-tiled (occupied + PAO), inner
// indices are the virtual/PNO axis restricted to one occupied-index-tuple's
// own (small) PNO count. See build_tot_array() in ta_builder.h.
//
// Inner-tile type is compile-time selectable (performance-parity: unlock
// multithreading). Default `TA::ArenaTensor<double>` matches MPQC's
// production layout (arena-pinned, SIMD-slab-packed inner cells) but its
// cells are NON-OWNING views into an arena page whose lifetime is the outer
// tile's shared_ptr, freed cross-thread by MADNESS lazy deletion -> segfaults
// at MAD_NUM_THREADS>1. Define SPTC_OWNING_TOT
// to use plain owning `TA::Tensor<double>` inner cells (refcount their own
// storage; no cross-thread dangling) — numerically identical per TA's own
// tests (einsum.cpp arena_matches_owning), at the cost of the arena packing.
#ifdef SPTC_OWNING_TOT
using SPTC_ToT_Tile = TA::Tensor<TA::Tensor<double>>;
#else
using SPTC_ToT_Tile = TA::Tensor<TA::ArenaTensor<double>>;
#endif
using ArrayToT = TA::DistArray<SPTC_ToT_Tile, TA::SparsePolicy>;

// Loaded tensors for one molecule. Field names mirror those used in the
// equation expressions. All 11 leaves from mpqc4:traces/all_equations.txt,
// on-disk column order is occupied(i) before hole/PAO(m) before virtual/PNO(a)
// before RI-auxiliary(k) throughout (see mpqc4:traces/SPTC_BENCH_MAPPING.md):
//   g0     ↔ g_m_1_m_2_K_1.txt   (uocc, uocc, ri)
//   g1     ↔ g_i_1_i_2_K_1.txt   (occ, occ, ri)
//   g      ↔ g_i_1_m_1_K_1.txt   (occ, uocc, ri)        [spec calls this g2]
//   c1     ↔ C_m_1_a_1_i_1.txt    (occ, uocc, virtual)
//   c2     ↔ C_m_1_a_1_i_1_i_2.txt (occ, occ, uocc, virtual)
//   f_i_i  ↔ f_i_1_i_2.txt        (occ, occ)
//   f_i_m  ↔ f_i_1_m_1.txt        (occ, uocc)
//   f_m_m  ↔ f_m_1_m_2.txt        (uocc, uocc)
//   s_m_m  ↔ s_m_1_m_2.txt        (uocc, uocc)
//   t_i_a       ↔ t_i_1_a_1.txt        (occ, virtual)
//   t_i_i_a_a   ↔ t_i_1_i_2_a_1_a_2.txt (occ, occ, virtual, virtual)
//
// c1_tot/c2_tot/t_i_a_tot/t_i_i_a_a_tot are the SAME data as c1/c2/t_i_a/
// t_i_i_a_a above, loaded a second time as genuine tensor-of-tensor arrays
// (outer=i[,i,]m for C, outer=i[,i] for t; inner=a[,a] restricted per
// occupied-index-tuple). The codegen picks flat or ToT per-equation based
// on whether the equation's DAG is expressible under TiledArray's ToT
// constraints (see gen_ta_equations.py) — both representations of each
// leaf coexist so either path is available.
struct TATensors {
  TA::TSpArrayD g0;
  TA::TSpArrayD g1;
  TA::TSpArrayD g;
  TA::TSpArrayD c1;
  TA::TSpArrayD c2;
  TA::TSpArrayD f_i_i;
  TA::TSpArrayD f_i_m;
  TA::TSpArrayD f_m_m;
  TA::TSpArrayD s_m_m;
  TA::TSpArrayD t_i_a;
  TA::TSpArrayD t_i_i_a_a;

  ArrayToT c1_tot;
  ArrayToT c2_tot;
  ArrayToT t_i_a_tot;
  ArrayToT t_i_i_a_a_tot;
};

#endif  // SPTC_TA_TENSORS_H
