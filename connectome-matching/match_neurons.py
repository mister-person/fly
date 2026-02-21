#!/usr/bin/env python3
"""
match_neurons.py
----------------
Match neurons between the BANC (Brain And Nerve Cord, female) and MCNS
(Male CNS, Janelia) Drosophila connectome datasets using connectivity
fingerprinting seeded by known anchor neuron pairs.

Algorithm:
  1. Load "anchor" pairs: neurons known to correspond across datasets.
  2. For every unmatched neuron, compute a connectivity fingerprint:
       a vector of [synapses_to_anchor_1, synapses_from_anchor_1,
                    synapses_to_anchor_2, synapses_from_anchor_2, ...]
  3. Compare fingerprints across datasets with cosine similarity.
  4. Report top-k BANC candidates for each MCNS neuron (greedy) OR
     find the globally optimal 1-to-1 assignment (Hungarian algorithm).

Expected data files
-------------------
  banc_conn.csv   : pre_id, post_id, syn_count
  mcns_conn.csv   : pre_id, post_id, syn_count
  anchors.csv     : banc_id, mcns_id
  banc_labels.csv : neuron_id, label   (optional; for transferring labels)

All IDs should be integers or strings – just be consistent.

Usage
-----
  python match_neurons.py \\
      --banc-conn data/banc_conn.csv \\
      --mcns-conn data/mcns_conn.csv \\
      --anchors   data/anchors.csv \\
      --banc-labels data/banc_labels.csv \\
      --out       results/mcns_matches.csv \\
      --top-k 5

  # For 1-to-1 optimal matching (slower but globally optimal):
  python match_neurons.py ... --optimal

  # Restrict BANC side to only labeled (tarsus MN) neurons:
  python match_neurons.py ... --labeled-only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_table(path, **kwargs):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    return pd.read_csv(path, **kwargs)


def normalize_conn_columns(df):
    """Rename whatever connectivity columns are present to pre_id/post_id/syn_count."""
    rename = {}
    for col in df.columns:
        lc = col.lower().replace(" ", "_")
        if "pre" in lc and ("id" in lc or "body" in lc or "root" in lc):
            rename[col] = "pre_id"
        elif "post" in lc and ("id" in lc or "body" in lc or "root" in lc):
            rename[col] = "post_id"
        elif any(x in lc for x in ["weight", "count", "n_syn", "nsynapses", "excitatory_x_connectivity"]):
            rename[col] = "syn_count"
    df = df.rename(columns=rename)
    if "syn_count" not in df.columns:
        print("  [warn] no synapse-count column found; assuming weight=1 per edge")
        df["syn_count"] = 1
    required = {"pre_id", "post_id", "syn_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Connectivity table missing columns: {missing}. "
                         f"Got: {list(df.columns)}")
    return df[["pre_id", "post_id", "syn_count"]]


def load_connectivity(path):
    print(f"  Loading connectivity: {path}")
    df = load_table(path)
    df = normalize_conn_columns(df)
    df["syn_count"] = pd.to_numeric(df["syn_count"], errors="coerce").fillna(1)
    print(f"    {len(df):,} edges, "
          f"{df['pre_id'].nunique():,} pre-neurons, "
          f"{df['post_id'].nunique():,} post-neurons")
    return df


def load_anchors(path):
    print(f"  Loading anchors: {path}")
    df = load_table(path)
    # Flexible column names
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if "banc" in lc:
            rename[col] = "banc_id"
        elif "mcns" in lc or "male" in lc or "janelia" in lc:
            rename[col] = "mcns_id"
    df = df.rename(columns=rename)
    if "banc_id" not in df.columns or "mcns_id" not in df.columns:
        raise ValueError(f"Anchors table must have banc_id and mcns_id columns. "
                         f"Got: {list(df.columns)}")
    df = df.dropna(subset=["banc_id", "mcns_id"])
    print(f"    {len(df):,} anchor pairs")
    return df


def load_labels(path):
    """Load label table → dict {neuron_id: label}."""
    print(f"  Loading labels: {path}")
    df = load_table(path)
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if "id" in lc and "neuron" not in lc:
            rename[col] = "neuron_id"
        elif "label" in lc or "type" in lc or "name" in lc or "class" in lc:
            rename[col] = "label"
    df = df.rename(columns=rename)
    if "neuron_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "neuron_id", df.columns[1]: "label"})
    label_map = dict(zip(df["neuron_id"], df["label"]))
    print(f"    {len(label_map):,} labeled neurons")
    return label_map


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprinting
# ─────────────────────────────────────────────────────────────────────────────

def build_adj(conn_df):
    """Build nested dict: adj[pre][post] = total syn_count."""
    adj = {}
    for _, row in conn_df.iterrows():
        pre, post, w = row["pre_id"], row["post_id"], row["syn_count"]
        adj.setdefault(pre, {})[post] = adj.get(pre, {}).get(post, 0) + w
    return adj


def compute_fingerprints(conn_df, query_ids, anchor_ids, log2_counts=True):
    """
    For each query neuron, compute a 2*|anchors|-dimensional fingerprint:
        [out_to_anchor_0, ..., out_to_anchor_n,
         in_from_anchor_0, ..., in_from_anchor_n]

    Parameters
    ----------
    conn_df   : DataFrame(pre_id, post_id, syn_count)
    query_ids : list of neuron IDs to fingerprint
    anchor_ids: ordered list of anchor neuron IDs
    log2_counts: apply log2(1+x) to smooth heavy-synapse dominance

    Returns
    -------
    fp_matrix  : np.ndarray shape (n_query, 2*n_anchor)
    query_ids  : list (same order as rows)
    """
    query_ids = list(query_ids)
    anchor_ids = list(anchor_ids)
    n_q = len(query_ids)
    n_a = len(anchor_ids)

    q_set = set(query_ids)
    a_set = set(anchor_ids)
    q_idx = {q: i for i, q in enumerate(query_ids)}
    a_idx = {a: i for i, a in enumerate(anchor_ids)}

    out_mat = np.zeros((n_q, n_a), dtype=np.float32)  # query → anchor
    in_mat  = np.zeros((n_q, n_a), dtype=np.float32)  # anchor → query

    for _, row in conn_df.iterrows():
        pre, post, w = row["pre_id"], row["post_id"], float(row["syn_count"])
        if pre in q_set and post in a_set:
            out_mat[q_idx[pre], a_idx[post]] += w
        if pre in a_set and post in q_set:
            in_mat[q_idx[post], a_idx[pre]] += w

    fp = np.hstack([out_mat, in_mat])  # (n_q, 2*n_a)

    if log2_counts:
        fp = np.log1p(fp)

    return fp, query_ids


# ─────────────────────────────────────────────────────────────────────────────
# Similarity & matching
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(A, B):
    """
    A: (m, d), B: (n, d) → returns (m, n) cosine similarity matrix.
    Clips to [-1, 1] for numerical safety.
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    sim = A_norm @ B_norm.T
    return np.clip(sim, -1.0, 1.0)


def match_greedy(sim, mcns_ids, banc_ids, top_k, label_map):
    """Return top-k BANC matches for each MCNS neuron."""
    records = []
    top_k = min(top_k, sim.shape[1])
    for i, mcns_id in enumerate(mcns_ids):
        top_j = np.argsort(sim[i])[::-1][:top_k]
        for rank, j in enumerate(top_j):
            banc_id = banc_ids[j]
            records.append({
                "mcns_id":    mcns_id,
                "rank":       rank + 1,
                "banc_match": banc_id,
                "similarity": float(sim[i, j]),
                "label":      label_map.get(banc_id) if label_map else None,
            })
    return pd.DataFrame(records)


def match_optimal(sim, mcns_ids, banc_ids, label_map):
    """
    1-to-1 optimal assignment via Hungarian algorithm.
    Maximises total cosine similarity.
    """
    print("  Running Hungarian algorithm for optimal 1-to-1 assignment "
          f"({len(mcns_ids)} × {len(banc_ids)})...")
    row_ind, col_ind = linear_sum_assignment(-sim)  # minimise negative sim
    records = []
    for ri, ci in zip(row_ind, col_ind):
        banc_id = banc_ids[ci]
        records.append({
            "mcns_id":    mcns_ids[ri],
            "banc_match": banc_id,
            "similarity": float(sim[ri, ci]),
            "label":      label_map.get(banc_id) if label_map else None,
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Anchor validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_anchors(banc_conn, mcns_conn, anchors, label_map=None):
    """
    Leave-one-out check: for each anchor, pretend it's unknown and see what
    rank the correct match gets.  A sanity check on data quality + method.
    """
    print("\n── Anchor leave-one-out validation ──")
    banc_anchors = anchors["banc_id"].tolist()
    mcns_anchors = anchors["mcns_id"].tolist()

    ranks = []
    for leave_out_idx in range(len(banc_anchors)):
        reduced_banc = banc_anchors[:leave_out_idx] + banc_anchors[leave_out_idx + 1:]
        reduced_mcns = mcns_anchors[:leave_out_idx] + mcns_anchors[leave_out_idx + 1:]

        if not reduced_banc:
            continue

        # Fingerprint the left-out neurons against remaining anchors
        banc_fp, _ = compute_fingerprints(
            banc_conn, [banc_anchors[leave_out_idx]], reduced_banc)
        mcns_fp, _ = compute_fingerprints(
            mcns_conn, [mcns_anchors[leave_out_idx]], reduced_mcns)

        # Compare against all anchor neurons in BANC
        all_banc_fp, all_banc_ids = compute_fingerprints(
            banc_conn, reduced_banc, reduced_banc)

        sim = cosine_similarity(mcns_fp, all_banc_fp)  # (1, n-1)
        ranked = np.argsort(sim[0])[::-1]
        correct_id = banc_anchors[leave_out_idx]
        try:
            rank = list(all_banc_ids[r] for r in ranked).index(correct_id) + 1
        except ValueError:
            rank = None

        ranks.append(rank)

    if ranks:
        valid_ranks = [r for r in ranks if r is not None]
        print(f"  Anchors tested : {len(ranks)}")
        if valid_ranks:
            print(f"  Mean rank      : {np.mean(valid_ranks):.2f}")
            print(f"  Median rank    : {np.median(valid_ranks):.1f}")
            print(f"  Rank=1 (exact) : {sum(r == 1 for r in valid_ranks)} "
                  f"/ {len(valid_ranks)}")
    else:
        print("  Not enough anchors to validate.")


# ─────────────────────────────────────────────────────────────────────────────
# API loaders (optional, requires packages)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_banc_connectivity(token=None):
    """
    Fetch BANC connectivity via the `banc` Python package.
    Install: pip install banc

    Returns DataFrame(pre_id, post_id, syn_count)
    """
    try:
        import banc  # noqa
    except ImportError:
        raise ImportError("pip install banc")

    # banc uses caveclient under the hood
    try:
        from caveclient import CAVEclient
    except ImportError:
        raise ImportError("pip install caveclient")

    client = CAVEclient("brain_and_nerve_cord")
    if token:
        client.auth.save_token(token=token)

    # Fetch all synapses (may be large; filter as needed)
    print("  Fetching BANC synapse table from cave...")
    syn = client.materialize.query_table("synapses_nt_v1",
                                         select_columns=["pre_pt_root_id",
                                                         "post_pt_root_id"])
    syn = syn.rename(columns={
        "pre_pt_root_id":  "pre_id",
        "post_pt_root_id": "post_id",
    })
    syn["syn_count"] = 1
    # Aggregate per pair
    conn = syn.groupby(["pre_id", "post_id"])["syn_count"].sum().reset_index()
    return conn


def fetch_mcns_connectivity(server="https://neuprint.janelia.org",
                             dataset="cns", token=None):
    """
    Fetch MCNS connectivity via neuprint-python.
    Install: pip install neuprint-python

    Returns DataFrame(pre_id, post_id, syn_count)
    """
    try:
        from neuprint import Client, fetch_synapses, NeuronCriteria as NC
    except ImportError:
        raise ImportError("pip install neuprint-python")

    c = Client(server, dataset=dataset, token=token)

    # Fetch all connections (this is big; in practice filter by ROI or cell type)
    print("  Fetching MCNS connectivity from neuprint...")
    conn, _ = fetch_synapses(NC(status="Traced"), NC(status="Traced"),
                              client=c)
    conn = conn.rename(columns={
        "bodyId_pre":  "pre_id",
        "bodyId_post": "post_id",
        "weight":      "syn_count",
    })
    return conn[["pre_id", "post_id", "syn_count"]]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Match BANC ↔ MCNS neurons via connectivity fingerprinting")
    p.add_argument("--banc-conn",   required=True,
                   help="BANC connectivity CSV/parquet (pre_id, post_id, syn_count)")
    p.add_argument("--mcns-conn",   required=True,
                   help="MCNS connectivity CSV/parquet (pre_id, post_id, syn_count)")
    p.add_argument("--anchors",     required=True,
                   help="Known anchor pairs CSV (banc_id, mcns_id)")
    p.add_argument("--banc-labels", default=None,
                   help="Optional label table CSV (neuron_id, label)")
    p.add_argument("--out",         default="mcns_matches.csv",
                   help="Output CSV path")
    p.add_argument("--top-k",       type=int, default=5,
                   help="Top-k BANC candidates per MCNS neuron (greedy mode)")
    p.add_argument("--optimal",     action="store_true",
                   help="Use Hungarian algorithm for 1-to-1 optimal assignment")
    p.add_argument("--labeled-only", action="store_true",
                   help="Only match MCNS neurons against labeled BANC neurons")
    p.add_argument("--validate",    action="store_true",
                   help="Run leave-one-out validation on anchors before matching")
    p.add_argument("--min-synapses", type=int, default=0,
                   help="Ignore edges with fewer than this many synapses")
    p.add_argument("--no-log",      action="store_true",
                   help="Disable log1p smoothing of synapse counts")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n═══════════════════════════════════════════")
    print("  BANC ↔ MCNS Neuron Matching")
    print("═══════════════════════════════════════════\n")

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data…")
    banc_conn = load_connectivity(args.banc_conn)
    mcns_conn = load_connectivity(args.mcns_conn)
    anchors   = load_anchors(args.anchors)

    label_map = {}
    if args.banc_labels:
        label_map = load_labels(args.banc_labels)

    # Filter low-synapse edges
    if args.min_synapses > 1:
        before = len(banc_conn)
        banc_conn = banc_conn[banc_conn["syn_count"] >= args.min_synapses]
        mcns_conn = mcns_conn[mcns_conn["syn_count"] >= args.min_synapses]
        print(f"  Filtered to ≥{args.min_synapses} synapses: "
              f"{before:,} → {len(banc_conn):,} BANC edges, "
              f"{len(mcns_conn):,} MCNS edges")

    # ── Anchor sets ────────────────────────────────────────────────────────
    # Only keep anchors that appear in the connectivity data (sanity)
    banc_conn_ids = set(banc_conn["pre_id"]) | set(banc_conn["post_id"])
    mcns_conn_ids = set(mcns_conn["pre_id"]) | set(mcns_conn["post_id"])

    n_before = len(anchors)
    print(len(banc_conn_ids))
    print(len(mcns_conn_ids))
    print(anchors[anchors["banc_id"].isin(banc_conn_ids) & anchors["mcns_id"].isin(mcns_conn_ids)])
    print(anchors["banc_id"].isin(banc_conn_ids) , anchors["mcns_id"].isin(mcns_conn_ids))
    anchors = anchors[
        anchors["banc_id"].isin(banc_conn_ids) &
        anchors["mcns_id"].isin(mcns_conn_ids)
    ].reset_index(drop=True)
    if len(anchors) < n_before:
        print(f"  [warn] {n_before - len(anchors)} anchors dropped "
              f"(not found in connectivity data)")
        print(anchors)
    if len(anchors) < 2:
        print("ERROR: Need at least 2 anchors. Check your data.")
        sys.exit(1)

    banc_anchor_set = set(anchors["banc_id"])
    mcns_anchor_set = set(anchors["mcns_id"])

    # ── Optional validation ────────────────────────────────────────────────
    if args.validate:
        validate_anchors(banc_conn, mcns_conn, anchors, label_map)

    # ── Query neurons ──────────────────────────────────────────────────────
    print("\nPreparing query sets…")
    if args.labeled_only and label_map:
        banc_query = [n for n in banc_conn_ids
                      if n in label_map and n not in banc_anchor_set]
        print(f"  BANC query (labeled only): {len(banc_query):,}")
    else:
        banc_query = [n for n in banc_conn_ids if n not in banc_anchor_set]
        print(f"  BANC query (all): {len(banc_query):,}")

    mcns_query = [n for n in mcns_conn_ids if n not in mcns_anchor_set]
    print(f"  MCNS query (all): {len(mcns_query):,}")

    if not banc_query:
        print("No BANC query neurons found. "
              "Check --labeled-only flag or that labels match connectivity IDs.")
        sys.exit(1)
    if not mcns_query:
        print("No MCNS query neurons found.")
        sys.exit(1)

    # ── Fingerprints ───────────────────────────────────────────────────────
    print("\nComputing fingerprints…")
    log2 = not args.no_log
    print(f"  log1p smoothing: {'on' if log2 else 'off'}")

    banc_fp, banc_ids = compute_fingerprints(
        banc_conn, banc_query, anchors["banc_id"].tolist(), log2_counts=log2)
    mcns_fp, mcns_ids = compute_fingerprints(
        mcns_conn, mcns_query, anchors["mcns_id"].tolist(), log2_counts=log2)

    # Drop neurons with zero fingerprint (no connection to any anchor)
    banc_nz = np.where(banc_fp.sum(axis=1) > 0)[0]
    mcns_nz = np.where(mcns_fp.sum(axis=1) > 0)[0]
    n_banc_zero = len(banc_ids) - len(banc_nz)
    n_mcns_zero = len(mcns_ids) - len(mcns_nz)
    if n_banc_zero or n_mcns_zero:
        print(f"  [warn] {n_banc_zero} BANC and {n_mcns_zero} MCNS neurons have "
              f"no connections to any anchor – skipping them")
    banc_fp   = banc_fp[banc_nz]
    banc_ids  = [banc_ids[i] for i in banc_nz]
    mcns_fp   = mcns_fp[mcns_nz]
    mcns_ids  = [mcns_ids[i] for i in mcns_nz]

    print(f"  Effective query: {len(banc_ids):,} BANC × {len(mcns_ids):,} MCNS "
          f"→ fingerprint dim {banc_fp.shape[1]}")

    # ── Similarity ─────────────────────────────────────────────────────────
    print("\nComputing cosine similarity matrix…")
    sim = cosine_similarity(mcns_fp, banc_fp)  # (n_mcns, n_banc)
    print(f"  Similarity matrix: {sim.shape}  "
          f"[min={sim.min():.3f}, mean={sim.mean():.3f}, max={sim.max():.3f}]")

    # ── Match ──────────────────────────────────────────────────────────────
    print("\nMatching…")
    if args.optimal:
        results = match_optimal(sim, mcns_ids, banc_ids, label_map)
    else:
        results = match_greedy(sim, mcns_ids, banc_ids, args.top_k, label_map)

    # ── Output ─────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\n✓ Wrote {len(results):,} rows → {out_path}")

    # Summary
    if label_map and "label" in results.columns:
        labeled_matches = results[results["label"].notna()]
        if not labeled_matches.empty:
            print(f"\nLabeled BANC neurons matched to MCNS neurons:")
            if "rank" in results.columns:
                top1 = results[results["rank"] == 1]
                top1_labeled = top1[top1["label"].notna()]
                print(top1_labeled[["mcns_id", "banc_match", "similarity", "label"]]
                      .sort_values("similarity", ascending=False)
                      .to_string(index=False))
            else:
                print(labeled_matches[["mcns_id", "banc_match", "similarity", "label"]]
                      .sort_values("similarity", ascending=False)
                      .to_string(index=False))

    print("\nDone.")
    return results


if __name__ == "__main__":
    main()
