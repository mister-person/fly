"""Systematic test of the voltage-target gradient (create + suppress) on many
simple readout-neuron cases, before trying recurrence.

Setup: one output neuron, N independent pulse-inputs (one input per pulse, spaced
well beyond the refractory period so pulses don't interfere).  A "case" picks
which pulses should fire (targets) and an initial weight pattern.  We train with
the voltage-target objective and check the OUTCOME by pulse selection:

  PASS iff  every target pulse fires exactly once in its window
        AND every non-target pulse is silent in its window
        AND there are no spikes outside any pulse window.

This directly tests spike-count / selection control (the method's job), without
depending on exact crossing times.  We also report the mean timing error.

Categories: create-from-dead, suppress-extras, mixed, full count sweep,
reproduce-a-known-pattern, many inputs, and coincident (credit-shared) inputs.
"""

import sys, os, types
sys.path.insert(0, "/workspace/project")
os.environ.setdefault("MPLBACKEND", "Agg")
for _n, _a in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        for _k, _v in _a.items():
            setattr(_m, _k, _v)
        sys.modules[_n] = _m

import numpy as np
from grad_method import lif_tangent, TH
from grad_multi_neuron import voltage_grad, train

SPACING = 100          # pulse spacing (>> refractory 22)
WLEN    = 95           # a pulse's "window" = [onset, onset+WLEN]
REFW    = 800.0        # reference weight to estimate a pulse's crossing time


def onsets_for(n):
    return [15 + SPACING * i for i in range(n)]


def crossing_time(onsets, i, T, w=REFW):
    """Natural crossing time of pulse i alone at weight w."""
    ia = np.zeros((len(onsets), T), bool); ia[i, onsets[i]] = True
    wv = np.zeros(len(onsets)); wv[i] = w
    sp = lif_tangent(wv, ia, T)[1]
    return sp[0] if sp else onsets[i] + 60


def pulse_of(t, onsets):
    for i, o in enumerate(onsets):
        if o <= t <= o + WLEN:
            return i
    return None


def evaluate(spikes, target_pulses, onsets):
    """Return (passed, detail). Checks pulse selection."""
    fired = {}
    for t in spikes:
        p = pulse_of(t, onsets)
        if p is None:
            return False, f"spike at {t} outside any pulse window"
        fired[p] = fired.get(p, 0) + 1
    for p in target_pulses:
        if fired.get(p, 0) != 1:
            return False, f"target pulse {p} fired {fired.get(p,0)}x (want 1)"
    for p in range(len(onsets)):
        if p not in target_pulses and fired.get(p, 0) != 0:
            return False, f"non-target pulse {p} fired {fired.get(p,0)}x (want 0)"
    return True, "ok"


def run_case(n_pulses, target_pulses, init_weights, T, ia=None, n_in=None):
    onsets = onsets_for(n_pulses)
    if ia is None:
        ia = np.zeros((n_pulses, T), bool)
        for i, o in enumerate(onsets):
            ia[i, o] = True
        n_in = n_pulses
    targets = [crossing_time(onsets, i, T) for i in target_pulses]
    w = train(np.array(init_weights, float), ia, targets, T, suppress=True,
              step=4.0, iters=350)
    spikes = lif_tangent(w, ia, T)[1]
    ok, detail = evaluate(spikes, target_pulses, onsets)
    # timing error vs reference crossing times (matched in order)
    terr = np.nan
    if len(spikes) == len(targets) and targets:
        terr = float(np.mean([abs(s - t) for s, t in zip(sorted(spikes), sorted(targets))]))
    return ok, detail, spikes, targets, terr


def main():
    rng = np.random.default_rng(0)
    results = []   # (category, name, ok, detail, terr)

    def add(cat, name, ok, detail, terr):
        results.append((cat, name, ok, detail, terr))
        flag = "PASS" if ok else "FAIL"
        te = f"{terr:5.1f}" if terr == terr else "  -  "
        print(f"  [{flag}] {cat:14s} {name:28s} terr={te}  {'' if ok else detail}")

    T4, T6 = 4 * SPACING + 120, 6 * SPACING + 120

    print("── create from dead (init all sub-threshold) ─────────────────────────")
    for k in range(1, 5):
        for seed in range(3):
            tp = sorted(rng.choice(4, k, replace=False).tolist())
            ok, d, sp, tg, te = run_case(4, tp, [100.0] * 4, T4)
            add("create", f"k={k} pulses={tp} s{seed}", ok, d, te)

    print("── suppress extras (init all fire) ───────────────────────────────────")
    for k in range(0, 4):
        for seed in range(3):
            tp = sorted(rng.choice(4, k, replace=False).tolist()) if k else []
            ok, d, sp, tg, te = run_case(4, tp, [700.0] * 4, T4)
            add("suppress", f"keep={tp} s{seed}", ok, d, te)

    print("── mixed (random hi/lo init, random target) ──────────────────────────")
    for seed in range(8):
        init = [rng.choice([100.0, 700.0]) for _ in range(4)]
        tp = sorted(rng.choice(4, rng.integers(1, 4), replace=False).tolist())
        ok, d, sp, tg, te = run_case(4, tp, init, T4)
        add("mixed", f"tp={tp} s{seed}", ok, d, te)

    print("── full count sweep 0..4 (init all fire) ─────────────────────────────")
    for k in range(0, 5):
        tp = list(range(k))
        ok, d, sp, tg, te = run_case(4, tp, [700.0] * 4, T4)
        add("count", f"count={k}", ok, d, te)

    print("── reproduce a known random pattern (feasible by construction) ───────")
    for seed in range(6):
        onsets = onsets_for(4)
        w_true = np.array([rng.uniform(500, 900) if rng.random() < .6 else 100.0
                           for _ in range(4)])
        ia = np.zeros((4, T4), bool)
        for i, o in enumerate(onsets):
            ia[i, o] = True
        tp = [i for i in range(4) if w_true[i] > 445]
        if not tp:
            continue
        init = [rng.uniform(100, 700) for _ in range(4)]
        ok, d, sp, tg, te = run_case(4, tp, init, T4, ia=ia, n_in=4)
        add("reproduce", f"tp={tp} s{seed}", ok, d, te)

    print("── many inputs (6 pulses) ────────────────────────────────────────────")
    for seed in range(5):
        tp = sorted(rng.choice(6, rng.integers(2, 5), replace=False).tolist())
        ok, d, sp, tg, te = run_case(6, tp, [rng.choice([100.0, 700.0]) for _ in range(6)], T6)
        add("many", f"tp={tp} s{seed}", ok, d, te)

    print("── coincident inputs (2 inputs share each pulse: credit-shared) ──────")
    for seed in range(5):
        onsets = onsets_for(4)
        n_in = 8
        ia = np.zeros((n_in, T4), bool)
        for i, o in enumerate(onsets):
            ia[2 * i, o] = True
            ia[2 * i + 1, o] = True        # two inputs at the same time
        tp = sorted(rng.choice(4, rng.integers(1, 4), replace=False).tolist())
        # init: pulses' pair-weights; sub-threshold total for non-firing
        init = np.full(n_in, 150.0)
        targets = [crossing_time(onsets, i, T4) for i in tp]
        w = train(init, ia, targets, T4, suppress=True, step=4.0, iters=350)
        spikes = lif_tangent(w, ia, T4)[1]
        ok, d = evaluate(spikes, tp, onsets)
        te = float(np.mean([abs(s - t) for s, t in zip(sorted(spikes), sorted(targets))])) \
            if len(spikes) == len(targets) and targets else np.nan
        add("coincident", f"tp={tp} s{seed}", ok, d, te)

    print("── EDGE / stress cases (expected to probe limits) ────────────────────")
    # (e1) pulses spaced 45 apart (< refractory+PSP): should interfere/fail
    for seed in range(3):
        close_onsets = [15 + 45 * i for i in range(4)]
        Tc = close_onsets[-1] + 160
        ia = np.zeros((4, Tc), bool)
        for i, o in enumerate(close_onsets):
            ia[i, o] = True
        tp = sorted(rng.choice(4, 2, replace=False).tolist())
        tgt = []
        for i in tp:
            wv = np.zeros(4); wv[i] = REFW
            s = lif_tangent(wv, ia, Tc)[1]
            tgt.append(s[0] if s else close_onsets[i] + 60)
        w = train(np.array([100.0] * 4), ia, tgt, Tc, suppress=True, step=4, iters=350)
        sp = lif_tangent(w, ia, Tc)[1]
        fired = {}
        for t in sp:
            for i, o in enumerate(close_onsets):
                if o <= t <= o + WLEN:
                    fired[i] = fired.get(i, 0) + 1
        ok = (all(fired.get(i, 0) == 1 for i in tp)
              and all(fired.get(i, 0) == 0 for i in range(4) if i not in tp)
              and sum(fired.values()) == len(sp))
        add("edge-close", f"spacing45 tp={tp} s{seed}", ok, f"got {sp}", np.nan)
    # (e2) impossible: ask ONE pulse to fire twice (a pulse can only fire once)
    onsets = onsets_for(2)
    ia = np.zeros((2, T4), bool)
    for i, o in enumerate(onsets):
        ia[i, o] = True
    tvec = [crossing_time(onsets, 0, T4), crossing_time(onsets, 0, T4) + 15]
    w = train(np.array([100.0, 100.0]), ia, tvec, T4, suppress=True, step=4, iters=350)
    sp = lif_tangent(w, ia, T4)[1]
    ok = len(sp) == 2 and all(pulse_of(t, onsets) == 0 for t in sp)
    add("edge-impossible", "one pulse fire 2x", ok, f"got {sp} (want 2 in pulse 0)", np.nan)

    # ── timing-precision breakdown across all non-edge cases ──
    all_te = [te for c, _, ok, _, te in results if te == te and not c.startswith("edge")]
    if all_te:
        all_te = np.array(all_te)
        print(f"\ntiming error over passing cases: mean={all_te.mean():.1f}  "
              f"within 5 steps={100*np.mean(all_te<=5):.0f}%  "
              f"within 15={100*np.mean(all_te<=15):.0f}%  max={all_te.max():.0f}")

    # ── summary ──
    print("\n" + "=" * 62)
    cats = {}
    for cat, _, ok, _, te in results:
        c = cats.setdefault(cat, [0, 0, []])
        c[0] += int(ok); c[1] += 1
        if te == te:
            c[2].append(te)
    total_ok = sum(c[0] for c in cats.values())
    total = sum(c[1] for c in cats.values())
    print(f"{'category':14s} {'pass':>8s}  {'mean timing err':>16s}")
    for cat, (o, n, tes) in cats.items():
        mt = f"{np.mean(tes):.1f}" if tes else "-"
        print(f"{cat:14s} {o:3d}/{n:<3d}  {mt:>16s}")
    print("-" * 62)
    print(f"{'TOTAL':14s} {total_ok:3d}/{total:<3d}  ({100*total_ok/total:.0f}% pass)")
    fails = [(c, n, d) for c, n, ok, d, _ in results if not ok]
    if fails:
        print("\nfailures:")
        for c, n, d in fails:
            print(f"  {c}/{n}: {d}")


if __name__ == "__main__":
    main()
