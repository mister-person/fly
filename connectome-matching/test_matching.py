#!/usr/bin/env python3
"""
test_matching.py
Synthetic test: create two "connectomes" with known ground-truth correspondence,
run the matching, and verify accuracy.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from match_neurons import compute_fingerprints, cosine_similarity, match_greedy

RNG = np.random.default_rng(42)

def make_synthetic_connectomes(n_anchors=20, n_query=30, n_neurons=None,
                                noise_level=0.1):
    """
    Build two synthetic connectomes that share the same underlying structure
    with a bit of noise added (simulating inter-dataset variability).

    Returns
    -------
    banc_conn, mcns_conn : DataFrames(pre_id, post_id, syn_count)
    anchors              : DataFrame(banc_id, mcns_id)
    ground_truth         : dict banc_query_id → mcns_query_id
    """
    if n_neurons is None:
        n_neurons = n_anchors + n_query + 20  # extra background neurons

    # Assign IDs (BANC: 1000-series, MCNS: 2000-series so they don't overlap)
    all_ids = np.arange(n_neurons)
    anchor_ids  = all_ids[:n_anchors]
    query_ids   = all_ids[n_anchors:n_anchors + n_query]
    bg_ids      = all_ids[n_anchors + n_query:]

    banc_ids = anchor_ids * 1     # BANC namespace
    mcns_ids = anchor_ids + 1000  # MCNS namespace for anchors
    banc_query = query_ids * 1
    mcns_query = query_ids + 1000  # same underlying neurons, different IDs

    ground_truth = dict(zip(banc_query, mcns_query))

    def add_noise(w, noise):
        """Multiplicative lognormal noise."""
        return max(0, w * np.exp(RNG.normal(0, noise)))

    def make_edges(neurons_a, neurons_b, id_offset_b=0, noise=noise_level):
        """Random sparse connectivity from neurons_a to neurons_b."""
        rows = []
        for pre in neurons_a:
            # Connect to a random subset of B
            targets = RNG.choice(neurons_b, size=RNG.integers(1, 6), replace=False)
            for post in targets:
                w = int(add_noise(RNG.integers(3, 30), noise))
                if w > 0:
                    rows.append({"pre_id": int(pre),
                                 "post_id": int(post) + id_offset_b,
                                 "syn_count": w})
        return rows

    # Both datasets share the same structural connectivity matrix but use
    # different IDs and have independent noise realizations.
    edges_banc, edges_mcns = [], []

    # query → anchor connections
    for q_banc, q_mcns in zip(banc_query, mcns_query):
        targets = RNG.choice(anchor_ids, size=RNG.integers(2, 8), replace=False)
        for a in targets:
            base_w = RNG.integers(5, 40)
            edges_banc.append({"pre_id": int(q_banc),
                                "post_id": int(a),
                                "syn_count": int(add_noise(base_w, noise_level))})
            edges_mcns.append({"pre_id": int(q_mcns),
                                "post_id": int(a) + 1000,
                                "syn_count": int(add_noise(base_w, noise_level))})

    # anchor → query connections
    for q_banc, q_mcns in zip(banc_query, mcns_query):
        sources = RNG.choice(anchor_ids, size=RNG.integers(2, 8), replace=False)
        for a in sources:
            base_w = RNG.integers(5, 40)
            edges_banc.append({"pre_id": int(a),
                                "post_id": int(q_banc),
                                "syn_count": int(add_noise(base_w, noise_level))})
            edges_mcns.append({"pre_id": int(a) + 1000,
                                "post_id": int(q_mcns),
                                "syn_count": int(add_noise(base_w, noise_level))})

    # Background noise edges
    for _ in range(n_neurons * 3):
        pre  = int(RNG.choice(all_ids))
        post = int(RNG.choice(all_ids))
        if pre != post:
            w = int(RNG.integers(1, 5))
            edges_banc.append({"pre_id": pre, "post_id": post, "syn_count": w})
            edges_mcns.append({"pre_id": pre + 1000, "post_id": post + 1000, "syn_count": w})

    banc_conn = pd.DataFrame(edges_banc)
    mcns_conn = pd.DataFrame(edges_mcns)

    # Aggregate
    banc_conn = banc_conn.groupby(["pre_id", "post_id"])["syn_count"].sum().reset_index()
    mcns_conn = mcns_conn.groupby(["pre_id", "post_id"])["syn_count"].sum().reset_index()

    anchors = pd.DataFrame({
        "banc_id": banc_ids.tolist(),
        "mcns_id": (banc_ids + 1000).tolist(),
    })

    return banc_conn, mcns_conn, anchors, ground_truth


def run_test(noise_level=0.1, n_anchors=20, n_query=30):
    print(f"\n── Synthetic test  anchors={n_anchors}  query={n_query}  "
          f"noise={noise_level} ──")

    banc_conn, mcns_conn, anchors, gt = make_synthetic_connectomes(
        n_anchors=n_anchors, n_query=n_query, noise_level=noise_level)

    banc_query = [b for b in gt.keys()]
    mcns_query = [gt[b] for b in banc_query]  # in same order → known ground truth

    banc_fp, banc_ids = compute_fingerprints(
        banc_conn, banc_query, anchors["banc_id"].tolist())
    mcns_fp, mcns_ids = compute_fingerprints(
        mcns_conn, mcns_query, anchors["mcns_id"].tolist())

    sim = cosine_similarity(mcns_fp, banc_fp)  # (n_mcns, n_banc)

    # For each MCNS query, check rank of correct BANC match
    ranks = []
    for i, mcns_id in enumerate(mcns_ids):
        # ground truth: mcns_id = banc_id + 1000  →  banc_id = mcns_id - 1000
        correct_banc = mcns_id - 1000
        if correct_banc not in banc_ids:
            continue
        j_correct = banc_ids.index(correct_banc)
        sorted_j = np.argsort(sim[i])[::-1]
        rank = list(sorted_j).index(j_correct) + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    top1  = np.sum(ranks == 1)
    top3  = np.sum(ranks <= 3)
    top5  = np.sum(ranks <= 5)
    n     = len(ranks)

    print(f"  n_tested : {n}")
    print(f"  Top-1    : {top1}/{n}  ({100*top1/n:.1f}%)")
    print(f"  Top-3    : {top3}/{n}  ({100*top3/n:.1f}%)")
    print(f"  Top-5    : {top5}/{n}  ({100*top5/n:.1f}%)")
    print(f"  Mean rank: {ranks.mean():.2f}")
    print(f"  Sim @ correct: {np.mean([sim[i, banc_ids.index(mcns_ids[i]-1000)] for i in range(n)]):.3f}")

    return top1 / n


if __name__ == "__main__":
    print("═══════════════════════════════")
    print("  Connectivity Fingerprint Test")
    print("═══════════════════════════════")

    # Easy: low noise, many anchors
    acc1 = run_test(noise_level=0.1, n_anchors=30, n_query=40)

    # Medium: moderate noise
    acc2 = run_test(noise_level=0.3, n_anchors=20, n_query=30)

    # Hard: high noise, few anchors
    acc3 = run_test(noise_level=0.5, n_anchors=10, n_query=20)

    print("\n── Summary ──")
    print(f"  Low noise / many anchors  : Top-1 = {acc1*100:.1f}%")
    print(f"  Medium noise              : Top-1 = {acc2*100:.1f}%")
    print(f"  High noise / few anchors  : Top-1 = {acc3*100:.1f}%")

    if acc1 >= 0.8 and acc2 >= 0.6:
        print("\n✓ Algorithm looks good.")
    else:
        print("\n✗ Accuracy lower than expected – check fingerprinting or data.")
