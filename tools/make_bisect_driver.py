#!/usr/bin/env python3
import re, sys

NAME_TO_EXPR = {
    "C_μ̃_ap1": "ts.c1_tot", "C_ap1_μ̃": "ts.c1_tot",
    "C_μ̃_ap2": "ts.c2_tot", "C_ap2_μ̃": "ts.c2_tot",
    "f_i_i": "ts.f_i_i",
    "f_i_μ̃": "ts.f_i_m", "f_μ̃_i": "permute_ij_2(ts.f_i_m)",
    "f_μ̃_μ̃": "ts.f_m_m",
    "g_i_i_Κ": "ts.g1",
    "g_i_μ̃_Κ": "ts.g", "g_μ̃_i_Κ": "permute_ij_3(ts.g)",
    "g_μ̃_μ̃_Κ": "ts.g0",
    "s_μ̃_μ̃": "ts.s_m_m",
    "t_ap1_i": "ts.t_i_a_tot",
    "t_ap2_ap2_i_i": "ts.t_i_i_a_a_tot",
}

def main():
    r2_path = sys.argv[1]
    out_path = sys.argv[2]
    text = open(r2_path, encoding="utf-8").read()
    m = re.search(r"whole_t2_residual\((.*?)\)\s*\{", text, re.S)
    params = m.group(1)
    parts, depth, cur = [], 0, ""
    for ch in params:
        if ch == "<": depth += 1
        if ch == ">": depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip()); cur = ""
        else:
            cur += ch
    if cur.strip(): parts.append(cur.strip())
    names = [p.split()[-1].lstrip("&") for p in parts]

    call_args = []
    for n in names:
        if n not in NAME_TO_EXPR:
            print(f"ERROR: unknown param name {n!r}", file=sys.stderr)
            sys.exit(1)
        call_args.append(NAME_TO_EXPR[n])

    driver = f'''// AUTO-GENERATED bisection test driver -- DO NOT EDIT BY HAND.
#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "ta_dumper.h"
#include "ta_stage.h"
#include "ta_tensor_loader.h"
#include "ta_tensors.h"

extern "C" __attribute__((weak)) void MKL_Set_Num_Threads(int);

inline TA::TSpArrayD permute_ij_2(const TA::TSpArrayD& src) {{
  TA::TSpArrayD dst;
  dst("j,i") = src("i,j");
  return dst;
}}
inline TA::TSpArrayD permute_ij_3(const TA::TSpArrayD& src) {{
  TA::TSpArrayD dst;
  dst("j,i,k") = src("i,j,k");
  return dst;
}}

{text}

int main(int argc, char** argv) {{
  if (MKL_Set_Num_Threads) MKL_Set_Num_Threads(1);
  TA::World& world = TA_SCOPED_INITIALIZE(argc, argv);
  if (const char* v = std::getenv("SPTC_MAD_WAIT_POLICY")) {{
    std::string policy = v;
    int us = 0;
    if (const char* u = std::getenv("SPTC_MAD_WAIT_SLEEP_US")) us = std::atoi(u);
    if (policy == "yield") madness::threadpool_wait_policy(madness::WaitPolicy::Yield);
    else if (policy == "sleep") madness::threadpool_wait_policy(madness::WaitPolicy::Sleep, us);
  }}
  std::string data_dir = argv[1];
  TATensors ts = load_ta_tensors(world, data_dir);
  world.gop.fence();
  std::cout << std::setprecision(15);

  auto t0 = std::chrono::steady_clock::now();
  ArrayToT r = whole_t2_residual({", ".join(call_args)});
  world.gop.fence();
  auto t1 = std::chrono::steady_clock::now();
  ChecksumResult cs = ta_compute_checksum(world, r);
  std::cout << "whole_t2_residual BISECT checksum: nnz=" << cs.nnz
            << " sum=" << cs.sum << " sumsq=" << cs.sumsq
            << " max_abs=" << cs.max_abs
            << " wall_s=" << std::chrono::duration<double>(t1 - t0).count()
            << std::endl;
  return 0;
}}
'''
    open(out_path, "w", encoding="utf-8").write(driver)
    print(f"wrote {out_path} ({len(names)} params, matched OK)")

if __name__ == "__main__":
    main()
