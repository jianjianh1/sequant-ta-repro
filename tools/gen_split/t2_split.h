#ifndef T2_SPLIT_H
#define T2_SPLIT_H
#include <tiledarray.h>
#include "ta_tensors.h"
// Runs t-indep precompute once, then times `ntrials` t-dependent
// residuals; prints one CSV line per trial. Returns final nnz.
long run_warm_t2_bench(TA::World& world, const ArrayToT& C_μ̃_ap1, const TA::TSpArrayD& g_i_μ̃_Κ, const ArrayToT& t_ap1_i, const ArrayToT& C_ap2_μ̃, const ArrayToT& C_μ̃_ap2, const TA::TSpArrayD& g_μ̃_μ̃_Κ, const ArrayToT& t_ap2_ap2_i_i, const TA::TSpArrayD& s_μ̃_μ̃, const TA::TSpArrayD& g_i_i_Κ, const TA::TSpArrayD& f_i_μ̃, const TA::TSpArrayD& g_μ̃_i_Κ, const TA::TSpArrayD& f_i_i, const TA::TSpArrayD& f_μ̃_μ̃, int ntrials, int warmup);
#endif
