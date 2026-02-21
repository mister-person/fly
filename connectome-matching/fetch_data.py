#!/usr/bin/env python3
"""
fetch_data.py
-------------
Download connectivity and label data from BANC and MCNS APIs.
Run this once to produce the CSV files that match_neurons.py needs.

Requirements:
  pip install banc caveclient neuprint-python pandas tqdm

Credentials:
  BANC:  get a CAVE token at https://globalv1.flywire-daf.com/auth/api/v1/refresh_token
  MCNS:  get a neuprint token at https://neuprint.janelia.org  (Account → token)

  Export them:
    export CAVE_TOKEN=<your-banc-token>
    export NEUPRINT_TOKEN=<your-mcns-token>
"""

import os
import sys
import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


# ─────────────────────────────────────────────────────────────────────────────
# BANC (CAVE / CAVEclient)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_banc(out_dir: Path, token: str = None):
    """
    Fetch BANC connectivity and labels via CAVEclient.
    Uses the public BANC materialisation endpoint.
    """
    try:
        from caveclient import CAVEclient
    except ImportError:
        print("ERROR: pip install caveclient")
        sys.exit(1)

    token = token or os.environ.get("CAVE_TOKEN")
    if not token:
        print("ERROR: set CAVE_TOKEN env var or pass --banc-token")
        sys.exit(1)

    print("Connecting to BANC (brain_and_nerve_cord)…")
    client = CAVEclient("brain_and_nerve_cord")
    client.auth.save_token(token=token)

    # ── Connectivity ──────────────────────────────────────────────────────
    print("  Fetching synapse table (synapses_nt_v1) – this may take a while…")
    syn = client.materialize.query_table(
        "synapses_nt_v1",
        select_columns=["pre_pt_root_id", "post_pt_root_id", "ctr_pt_position"],
    )
    syn = syn.rename(columns={
        "pre_pt_root_id":  "pre_id",
        "post_pt_root_id": "post_id",
    })
    # Remove autapses and root-0 (unassigned)
    syn = syn[(syn["pre_id"] != 0) & (syn["post_id"] != 0)]
    syn = syn[syn["pre_id"] != syn["post_id"]]

    # Aggregate per pair
    conn = (syn.groupby(["pre_id", "post_id"])
               .size()
               .reset_index(name="syn_count"))
    out_conn = out_dir / "banc_conn.csv"
    conn.to_csv(out_conn, index=False)
    print(f"  ✓ BANC connectivity → {out_conn}  ({len(conn):,} pairs)")

    # ── Labels ─────────────────────────────────────────────────────────────
    print("  Fetching cell-type annotations…")
    try:
        ann = client.materialize.query_table("cell_info")
    except Exception:
        try:
            ann = client.materialize.query_table("nucleus_neuron_svm")
        except Exception as e:
            print(f"  [warn] Could not fetch labels: {e}")
            return

    # Normalize
    id_cols   = [c for c in ann.columns if "root" in c.lower()]
    type_cols = [c for c in ann.columns
                 if any(x in c.lower() for x in ["type", "label", "class", "cell"])]
    if id_cols and type_cols:
        labels = ann[[id_cols[0], type_cols[0]]].rename(
            columns={id_cols[0]: "neuron_id", type_cols[0]: "label"})
        out_labels = out_dir / "banc_labels.csv"
        labels.to_csv(out_labels, index=False)
        print(f"  ✓ BANC labels → {out_labels}  ({len(labels):,} neurons)")


# ─────────────────────────────────────────────────────────────────────────────
# MCNS (neuprint)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_mcns(out_dir: Path, token: str = None,
               server: str = "https://neuprint.janelia.org",
               dataset: str = "cns"):
    """
    Fetch MCNS connectivity and labels via neuprint-python.
    """
    try:
        from neuprint import Client, fetch_neurons, NeuronCriteria as NC
        import neuprint
    except ImportError:
        print("ERROR: pip install neuprint-python")
        sys.exit(1)

    token = token or os.environ.get("NEUPRINT_TOKEN")
    if not token:
        print("ERROR: set NEUPRINT_TOKEN env var or pass --mcns-token")
        sys.exit(1)

    print(f"Connecting to MCNS neuprint ({server}, dataset={dataset})…")
    client = Client(server, dataset=dataset, token=token)

    # ── Labels ─────────────────────────────────────────────────────────────
    print("  Fetching neuron metadata…")
    neuron_df, _ = fetch_neurons(NC(status="Traced"), client=client)
    id_col   = "bodyId"
    type_col = next((c for c in neuron_df.columns
                     if any(x in c.lower() for x in ["type", "class", "instance"])),
                    None)
    if type_col:
        labels = neuron_df[[id_col, type_col]].rename(
            columns={id_col: "neuron_id", type_col: "label"})
        out_labels = out_dir / "mcns_labels.csv"
        labels.to_csv(out_labels, index=False)
        print(f"  ✓ MCNS labels → {out_labels}  ({len(labels):,} neurons)")

    # ── Connectivity ──────────────────────────────────────────────────────
    # Fetch all traced→traced connections
    print("  Fetching connectivity (this can be large)…")
    try:
        from neuprint import fetch_adjacency
        adj_df, _ = fetch_adjacency(
            NC(status="Traced"), NC(status="Traced"), client=client)
        conn = adj_df.rename(columns={
            "bodyId_pre":  "pre_id",
            "bodyId_post": "post_id",
            "weight":      "syn_count",
        })[["pre_id", "post_id", "syn_count"]]
    except Exception:
        # Fallback: paginated synapse fetch
        from neuprint import fetch_synapse_connections
        syn = fetch_synapse_connections(
            NC(status="Traced"), NC(status="Traced"), client=client)
        syn = syn.rename(columns={
            "bodyId_pre":  "pre_id",
            "bodyId_post": "post_id",
        })
        conn = (syn.groupby(["pre_id", "post_id"])
                   .size()
                   .reset_index(name="syn_count"))

    out_conn = out_dir / "mcns_conn.csv"
    conn.to_csv(out_conn, index=False)
    print(f"  ✓ MCNS connectivity → {out_conn}  ({len(conn):,} pairs)")


# ─────────────────────────────────────────────────────────────────────────────
# Anchor table builder
# ─────────────────────────────────────────────────────────────────────────────

def build_anchor_table_from_shared_labels(banc_labels_path, mcns_labels_path,
                                           out_path, shared_types=None):
    """
    Build anchor pairs by matching neurons with identical cell-type labels
    in both datasets.  A label appearing exactly once in each dataset = anchor.

    shared_types: optional list of cell-type strings to restrict to.
    """
    banc = pd.read_csv(banc_labels_path).rename(
        columns={lambda c: c.lower().strip()})
    mcns = pd.read_csv(mcns_labels_path).rename(
        columns={lambda c: c.lower().strip()})

    # Normalize column names
    for df in [banc, mcns]:
        df.columns = [c.lower().strip() for c in df.columns]

    banc_id_col = next(c for c in banc.columns if "id" in c)
    mcns_id_col = next(c for c in mcns.columns if "id" in c)
    banc_lbl    = next(c for c in banc.columns if "label" in c or "type" in c)
    mcns_lbl    = next(c for c in mcns.columns if "label" in c or "type" in c)

    if shared_types:
        banc = banc[banc[banc_lbl].isin(shared_types)]
        mcns = mcns[mcns[mcns_lbl].isin(shared_types)]

    # Only use labels that appear exactly once in each dataset → unambiguous
    banc_unique = banc[banc_lbl].value_counts()
    mcns_unique = mcns[mcns_lbl].value_counts()
    unique_labels = set(banc_unique[banc_unique == 1].index) & \
                    set(mcns_unique[mcns_unique == 1].index)

    if not unique_labels:
        print("No unambiguous shared labels found. You may need to provide "
              "anchors manually.")
        return pd.DataFrame(columns=["banc_id", "mcns_id"])

    banc_filt = banc[banc[banc_lbl].isin(unique_labels)]
    mcns_filt = mcns[mcns[mcns_lbl].isin(unique_labels)]

    merged = pd.merge(
        banc_filt[[banc_id_col, banc_lbl]].rename(
            columns={banc_id_col: "banc_id", banc_lbl: "label"}),
        mcns_filt[[mcns_id_col, mcns_lbl]].rename(
            columns={mcns_id_col: "mcns_id", mcns_lbl: "label"}),
        on="label",
    )
    anchors = merged[["banc_id", "mcns_id", "label"]]
    anchors.to_csv(out_path, index=False)
    print(f"✓ Built {len(anchors)} anchor pairs → {out_path}")
    return anchors


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Download BANC/MCNS data")
    p.add_argument("--banc",        action="store_true", help="Fetch BANC data")
    p.add_argument("--mcns",        action="store_true", help="Fetch MCNS data")
    p.add_argument("--build-anchors", action="store_true",
                   help="Build anchor table from shared labels after fetching")
    p.add_argument("--banc-token",  default=None)
    p.add_argument("--mcns-token",  default=None)
    p.add_argument("--mcns-server", default="https://neuprint.janelia.org")
    p.add_argument("--mcns-dataset", default="cns")
    p.add_argument("--out-dir",     default=str(DATA_DIR))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.banc and not args.mcns and not args.build_anchors:
        print("Nothing to do. Use --banc, --mcns, and/or --build-anchors.")
        p.print_help()
        return

    if args.banc:
        fetch_banc(out_dir, token=args.banc_token)

    if args.mcns:
        fetch_mcns(out_dir, token=args.mcns_token,
                   server=args.mcns_server, dataset=args.mcns_dataset)

    if args.build_anchors:
        banc_lbl = out_dir / "banc_labels.csv"
        mcns_lbl = out_dir / "mcns_labels.csv"
        if not banc_lbl.exists() or not mcns_lbl.exists():
            print("ERROR: need both banc_labels.csv and mcns_labels.csv first. "
                  "Run with --banc and --mcns first.")
            sys.exit(1)
        build_anchor_table_from_shared_labels(
            banc_lbl, mcns_lbl,
            out_path=out_dir / "anchors.csv",
        )


if __name__ == "__main__":
    main()
