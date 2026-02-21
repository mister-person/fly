all the code in this folder was written by claude.

# BANC ↔ MCNS Neuron Matching

Match neurons between the **BANC** (female brain+nerve cord, Harvard/Lee lab) and
**MCNS** (male CNS, Janelia) Drosophila connectome datasets using connectivity
fingerprinting seeded by known anchor pairs.

## Problem

MCNS is missing tarsus motor neuron labels that BANC has. We know some neurons
are the same across both datasets (anchors). Use those anchors + connectivity
structure to infer which unlabeled MCNS neurons correspond to which labeled BANC neurons.

## Algorithm

```
For each neuron N in each dataset:
  fingerprint(N) = [synapses_N→anchor_0, synapses_anchor_0→N,
                    synapses_N→anchor_1, synapses_anchor_1→N, ...]
                    (log1p-smoothed)

For each MCNS neuron, find BANC neurons with most similar fingerprint
  → cosine similarity
  → report top-k matches (greedy) OR globally optimal 1-to-1 (Hungarian)
```

The more anchors you have, the higher-dimensional (and more discriminative)
the fingerprint becomes.

## Setup

```bash
cd ~/projects/connectome-matching
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy

# For API access (optional):
pip install banc caveclient neuprint-python
```

## Data files

| File | Columns | Notes |
|------|---------|-------|
| `data/banc_conn.csv` | pre_id, post_id, syn_count | BANC connectivity |
| `data/mcns_conn.csv` | pre_id, post_id, syn_count | MCNS connectivity |
| `data/anchors.csv` | banc_id, mcns_id | Known homologs |
| `data/banc_labels.csv` | neuron_id, label | Tarsus MN labels etc |

Column names are flexible — the script auto-detects `pre*id`, `post*id`, `syn*` etc.

## Workflow

### Option A: Download from APIs

```bash
source .venv/bin/activate

# Set credentials
export CAVE_TOKEN=<your-banc-token>        # from https://globalv1.flywire-daf.com/auth/api/v1/refresh_token
export NEUPRINT_TOKEN=<your-mcns-token>    # from https://neuprint.janelia.org

# Download both datasets + build anchor table from shared cell types
python fetch_data.py --banc --mcns --build-anchors
```

### Option B: Use existing data files

Put your CSVs in `data/` and skip `fetch_data.py`.

**Anchors** can come from:
- A manually curated spreadsheet of known homologs
- The shared cell-type approach (`fetch_data.py --build-anchors`)
- Cross-registration tools like NBLAST or bridging transforms

### Run the matching

```bash
source .venv/bin/activate

# Greedy top-5 matches per MCNS neuron, restricted to labeled BANC neurons
python match_neurons.py \
    --banc-conn   data/banc_conn.csv \
    --mcns-conn   data/mcns_conn.csv \
    --anchors     data/anchors.csv \
    --banc-labels data/banc_labels.csv \
    --out         results/mcns_matches.csv \
    --top-k 5 \
    --labeled-only

# 1-to-1 globally optimal assignment
python match_neurons.py ... --optimal

# Include anchor leave-one-out validation (sanity check)
python match_neurons.py ... --validate
```

## Output

`results/mcns_matches.csv`:

| mcns_id | rank | banc_match | similarity | label |
|---------|------|------------|------------|-------|
| 2001234 | 1 | 720575940… | 0.91 | TarsusMN_1 |
| 2001234 | 2 | 720575941… | 0.83 | TarsusMN_2 |
| … | | | | |

`similarity` is cosine similarity [−1, 1]; a good match is typically > 0.7.

## Confidence heuristics

- **High confidence**: similarity > 0.8, top-2 similarity gap > 0.15
- **Check manually**: similarity 0.5–0.8 or close second-place score
- **Low quality**: similarity < 0.5 (neuron may not connect to any anchor)

## Tuning

| Flag | Effect |
|------|--------|
| `--min-synapses 5` | Filter weak edges (reduces noise) |
| `--no-log` | Disable log1p smoothing (try if syn counts are already normalized) |
| `--labeled-only` | Only search within labeled BANC neurons (faster, focused) |
| More anchors | Better — add any neuron pair you can confidently identify |

## Test

```bash
source .venv/bin/activate
python test_matching.py
```

Should show >85% top-1 accuracy on the synthetic easy case.
