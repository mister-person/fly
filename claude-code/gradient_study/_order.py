"""Exact minimum-feedback-arc-set ordering.

A neuron ordering is only useful for reading a plot if it puts cause before effect, so the
right objective is: choose a linear order minimising the number of edges that point BACKWARDS.
That is minimum feedback arc set, NP-hard in general -- but these nets have N <= 26, and the
subset DP is O(2^N * N), so it is cheap to solve EXACTLY rather than settle for a greedy
heuristic whose backward count you then cannot interpret.

    dp[S] = min over v in S of  dp[S - v] + (edges from v into S - v)

placing v last in the prefix S makes exactly its edges into already-placed nodes backward.
"""
import numpy as np


def min_fas_order(E, N, fixed_first=None):
    """(order, n_backward).  fixed_first pins a node to position 0 (e.g. the input)."""
    out = [[] for _ in range(N)]
    for a, b in E:
        if int(a) != int(b):
            out[int(a)].append(int(b))
    full = (1 << N) - 1
    INF = float("inf")
    dp = np.full(1 << N, INF)
    ch = np.full(1 << N, -1, np.int8)
    dp[0] = 0.0
    for S in range(1 << N):
        if dp[S] == INF:
            continue
        base = dp[S]
        for v in range(N):
            bit = 1 << v
            if S & bit:
                continue
            if fixed_first is not None and S == 0 and v != fixed_first:
                continue
            back = 0
            for u in out[v]:
                if S & (1 << u):
                    back += 1
            ns = S | bit
            c = base + back
            if c < dp[ns]:
                dp[ns] = c
                ch[ns] = v
    order, S = [], full
    while S:
        v = int(ch[S])
        order.append(v)
        S ^= (1 << v)
    order.reverse()
    return order, int(dp[full])


def classify(E, order):
    """(forward, backward) edge index lists for a given order."""
    pos = {n: i for i, n in enumerate(order)}
    fwd, bwd = [], []
    for i, (a, b) in enumerate(E):
        (bwd if pos[int(a)] >= pos[int(b)] else fwd).append(i)
    return fwd, bwd
