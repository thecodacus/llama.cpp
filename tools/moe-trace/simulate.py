#!/usr/bin/env python3
"""Replay a moe-trace CSV against expert-cache policies.

Each layer gets an independent cache of S expert slots (experts are
layer-specific).  A "hit" means a routed expert was already resident in
VRAM when the router asked for it.  Hit rate directly bounds how much
decode-time RAM traffic a cache can remove.

Policies:
  static-oracle   top-S experts per layer by whole-trace frequency (upper bound)
  static-prefill  top-S by PREFILL routing only, fixed for decode (deployable v1)
  lru             evict least-recently-used on miss
  lfu-decay       score = EMA of use; evict lowest; insert always
  reelect-K       EMA counters, cache membership re-elected every K decode steps
                  (the Phase 2 design: batch eviction, no per-token churn)

Usage: simulate.py trace.csv [--budgets 0.125,0.25,0.5] [--reelect 32]
"""
import argparse
import csv
import math
import sys
from collections import defaultdict


def load(path):
    prefill, decode = defaultdict(list), defaultdict(list)  # layer -> [ids per step]
    n_expert_seen = 0
    with open(path) as f:
        for row in csv.reader(f):
            pos, layer, ids = int(row[0]), int(row[1]), [int(x) for x in row[2:]]
            (prefill if pos < 0 else decode)[layer].append((pos, ids))
            n_expert_seen = max(n_expert_seen, max(ids) + 1)
    for d in (prefill, decode):
        for layer in d:
            d[layer].sort(key=lambda t: t[0])
            d[layer] = [ids for _, ids in d[layer]]
    return prefill, decode, n_expert_seen


def sim_static(decode, resident):
    hits = total = 0
    for layer, steps in decode.items():
        r = resident.get(layer, set())
        for ids in steps:
            for e in ids:
                hits += e in r
                total += 1
    return hits / max(total, 1)


def top_by_freq(steps_by_layer, S):
    resident = {}
    for layer, steps in steps_by_layer.items():
        freq = defaultdict(int)
        for ids in steps:
            for e in ids:
                freq[e] += 1
        resident[layer] = set(sorted(freq, key=freq.get, reverse=True)[:S])
    return resident


def sim_lru(decode, S):
    hits = total = 0
    for layer, steps in decode.items():
        cache, clock = {}, 0
        for ids in steps:
            for e in ids:
                clock += 1
                if e in cache:
                    hits += 1
                else:
                    if len(cache) >= S:
                        cache.pop(min(cache, key=cache.get))
                    # inserted AFTER the miss (upload happens post-hoc)
                cache[e] = clock
                total += 1
    return hits / max(total, 1)


def sim_lfu_decay(decode, S, alpha=0.95):
    hits = total = 0
    for layer, steps in decode.items():
        score, cache = defaultdict(float), set()
        for ids in steps:
            for k in score:
                score[k] *= alpha
            for e in ids:
                score[e] += 1.0
                if e in cache:
                    hits += 1
                elif len(cache) < S:
                    cache.add(e)
                else:
                    victim = min(cache, key=lambda k: score[k])
                    if score[e] >= score[victim]:
                        cache.discard(victim)
                        cache.add(e)
                total += 1
    return hits / max(total, 1)


def sim_reelect(decode, prefill, S, K, alpha=0.98):
    hits = total = 0
    for layer, steps in decode.items():
        score = defaultdict(float)
        for ids in prefill.get(layer, []):        # free warm-up from prompt routing
            for e in ids:
                score[e] += 1.0
        cache = set(sorted(score, key=score.get, reverse=True)[:S])
        for step, ids in enumerate(steps):
            for e in ids:
                hits += e in cache
                total += 1
                score[e] += 1.0
            if step % K == K - 1:
                for k in score:
                    score[k] *= alpha
                cache = set(sorted(score, key=score.get, reverse=True)[:S])
    return hits / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--budgets", default="0.0625,0.125,0.25,0.5",
                    help="cache size as fraction of expert count")
    ap.add_argument("--reelect", type=int, default=32)
    args = ap.parse_args()

    prefill, decode, n_expert = load(args.trace)
    n_layers = len(decode)
    n_steps = max(len(s) for s in decode.values()) if decode else 0
    n_used = len(next(iter(decode.values()))[0]) if decode else 0
    print(f"trace: {n_layers} MoE layers, {n_expert} experts, "
          f"{n_used} active/token, {n_steps} decode steps\n")

    # skew snapshot: what share of routed traffic hits the top-k% experts
    freq = defaultdict(int)
    for steps in decode.values():
        for ids in steps:
            for e in ids:
                freq[e] += 1
    ranked = sorted(freq.values(), reverse=True)
    tot = sum(ranked)
    for frac in (0.1, 0.25, 0.5):
        k = max(1, int(len(ranked) * frac))
        print(f"top {frac:>4.0%} of (layer,expert) pairs carry "
              f"{sum(ranked[:k])/tot:.1%} of decode routing")
    print()

    hdr = f"{'budget':>8} {'slots':>6} | {'st-oracle':>9} {'st-prefill':>10} " \
          f"{'lru':>7} {'lfu-decay':>9} {'reelect-'+str(args.reelect):>10}"
    print(hdr)
    print("-" * len(hdr))
    for frac in [float(x) for x in args.budgets.split(",")]:
        S = max(1, int(n_expert * frac))
        oracle = sim_static(decode, top_by_freq(decode, S))
        st_pre = sim_static(decode, top_by_freq(prefill, S)) if prefill else float("nan")
        lru = sim_lru(decode, S)
        lfu = sim_lfu_decay(decode, S)
        re = sim_reelect(decode, prefill, S, args.reelect)
        print(f"{frac:>8.4f} {S:>6} | {oracle:>9.1%} {st_pre:>10.1%} "
              f"{lru:>7.1%} {lfu:>9.1%} {re:>10.1%}")


if __name__ == "__main__":
    main()
