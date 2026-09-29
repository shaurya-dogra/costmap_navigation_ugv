#!/usr/bin/env python3
"""
dstar_lite.py - incremental global planning  (SIH PS 26126)
===========================================================

D* Lite (Koenig & Likhachev, 2002, the optimised version) on the same 8-connected grid,
with the same cost model, as `costmap_prototype.astar`:

    step cost  = length_in_cells * (1 + plan_cost_weight * cell_cost / 255)   (entering)
    LETHAL     = impassable;  UNKNOWN = plan_unknown_cost, expensive but passable

WHY IT IS HERE
--------------
A* answers every replan from scratch. That is fine when nothing is remembered between
plans, but the global layer is exactly the situation D* Lite was designed for: the
goal is fixed, the robot moves, and the map changes a little at a time - a handful of
cells near the robot per frame, because that is all a camera with a few-metre horizon
can reveal. D* Lite searches BACKWARD from the goal and keeps that search; after a
change it repairs only the part of the cost-to-goal field the change actually affects,
and the robot moving costs nothing at all (the `km` key offset absorbs it). So a
replan that A* pays for in full is, most of the time, a few hundred vertex updates.

It is used for the GLOBAL planner only. The local planner's grid is rebuilt in the
robot frame every frame - every cell moves each time the robot does - so there is no
previous search to repair and A* is already the right tool there.

WHAT IS KEPT FROM THE A* PLANNER
--------------------------------
Same neighbours, same costs, same admissible heuristic family, so on the same grid the
two return paths of the same cost (asserted in test_nav.py). When the goal is NOT
reachable, D* Lite has no path to give; A*'s graceful "path to the closest expanded
cell" is what the Navigator wants then, so `plan()` reports `None` and the caller falls
back to A* for that case.
"""

from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np

INF = float("inf")
SQRT2 = math.sqrt(2.0)
_NB = ((-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2),
       (0, -1, 1.0), (0, 1, 1.0),
       (1, -1, SQRT2), (1, 0, 1.0), (1, 1, SQRT2))


_EPS = 1e-9


def _key_lt(a, b) -> bool:
    """Lexicographic key order with a tolerance on k1. The first component is a sum of
    path cost, heuristic and km accumulated in a different order for different vertices,
    so two keys that are EQUAL in exact arithmetic routinely differ by an ulp - and
    comparing them raw skips the k2 tie-break, ends the search early and leaves an
    under-consistent vertex on the path (caught by the incremental test in test_nav)."""
    if a[0] < b[0] - _EPS:
        return True
    if a[0] > b[0] + _EPS:
        return False
    return a[1] < b[1] - _EPS


def _octile(ax, ay, bx, by) -> float:
    dx, dy = abs(ax - bx), abs(ay - by)
    return max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)


class DStarLite:
    """
    One goal on one fixed-size grid. Build it once per goal, then call `update_costs()`
    with the new per-cell (cost, blocked) arrays and `plan(start)` on every replan.
    """

    def __init__(self, shape, goal, cost: np.ndarray, blocked: np.ndarray, cost_weight: float):
        self.nx, self.ny = int(shape[0]), int(shape[1])
        self.goal = (int(goal[0]), int(goal[1]))
        self.cost_weight = float(cost_weight)
        n = self.nx * self.ny
        self.g = [INF] * n
        self.rhs = [INF] * n
        self.key: list = [None] * n            # key currently in the heap, or None
        self.heap: list = []
        self.km = 0.0
        self.last_start: Optional[tuple] = None
        self.mult, self.blk = self._arrays(cost, blocked)
        self.expanded = 0                      # vertex expansions, for profiling
        gi = self.goal[0] * self.ny + self.goal[1]
        self.rhs[gi] = 0.0
        self._push(gi, self._calc_key(gi, self.goal))

    # -- grid ----------------------------------------------------------------------
    def _arrays(self, cost, blocked):
        mult = (1.0 + (self.cost_weight / 255.0) * np.asarray(cost, np.float64)).ravel().tolist()
        return mult, np.asarray(blocked, bool).ravel().tolist()

    def _nbrs(self, i):
        ix, iy = divmod(i, self.ny)
        for dx, dy, L in _NB:
            jx, jy = ix + dx, iy + dy
            if 0 <= jx < self.nx and 0 <= jy < self.ny:
                yield jx * self.ny + jy, L

    def _c(self, j, L):
        """Cost of stepping INTO cell j over an edge of length L."""
        return INF if self.blk[j] else L * self.mult[j]

    # -- queue ---------------------------------------------------------------------
    def _calc_key(self, i, start):
        m = min(self.g[i], self.rhs[i])
        ix, iy = divmod(i, self.ny)
        # k1 rounded so exact-arithmetic ties are ties in the heap order too
        return (round(m + _octile(start[0], start[1], ix, iy) + self.km, 9), m)

    def _push(self, i, k):
        self.key[i] = k
        heapq.heappush(self.heap, (k[0], k[1], i))

    def _top(self):
        """Smallest VALID heap entry (lazy deletion), or None."""
        h = self.heap
        while h:
            k0, k1, i = h[0]
            k = self.key[i]
            if k is not None and k[0] == k0 and k[1] == k1:
                return h[0]
            heapq.heappop(h)
        return None

    def _update_vertex(self, i, start):
        if self.g[i] != self.rhs[i]:
            self._push(i, self._calc_key(i, start))
        else:
            self.key[i] = None                 # drop from the queue (lazily)

    def _best_rhs(self, i):
        best = INF
        g, blk, mult = self.g, self.blk, self.mult
        for j, L in self._nbrs(i):
            if not blk[j]:
                v = L * mult[j] + g[j]
                if v < best:
                    best = v
        return best

    # -- the algorithm --------------------------------------------------------------
    def _compute(self, start, max_expansions: int):
        si = start[0] * self.ny + start[1]
        g, rhs = self.g, self.rhs
        n = 0
        while True:
            top = self._top()
            if top is None:
                break
            k_start = self._calc_key(si, start)
            if not _key_lt((top[0], top[1]), k_start) and rhs[si] == g[si]:
                break
            n += 1
            if n > max_expansions:
                break
            k_old = (top[0], top[1])
            u = top[2]
            k_new = self._calc_key(u, start)
            if _key_lt(k_old, k_new):
                self._push(u, k_new)
                continue
            heapq.heappop(self.heap)
            self.key[u] = None
            if g[u] > rhs[u]:
                g[u] = rhs[u]
                gu = g[u]
                cu = self.mult[u]
                if not self.blk[u]:
                    for s, L in self._nbrs(u):
                        v = L * cu + gu        # s -> u enters u
                        if v < rhs[s]:
                            rhs[s] = v
                            self._update_vertex(s, start)
            else:
                g_old = g[u]
                g[u] = INF
                gi = self.goal[0] * self.ny + self.goal[1]
                cu = self.mult[u]
                for s, L in list(self._nbrs(u)) + [(u, 0.0)]:
                    if s == gi:
                        continue
                    via_u = INF if (s == u or self.blk[u]) else L * cu + g_old
                    if rhs[s] == via_u or s == u:
                        rhs[s] = self._best_rhs(s)
                    self._update_vertex(s, start)
        self.expanded += n

    # -- public ----------------------------------------------------------------------
    def update_costs(self, cost: np.ndarray, blocked: np.ndarray, start) -> int:
        """
        Swap in a new (cost, blocked) grid. Only cells that actually changed are
        processed: a changed cell alters the cost of every edge INTO it, so each of its
        neighbours has its rhs recomputed. Returns how many cells changed.
        """
        mult, blk = self._arrays(cost, blocked)
        old_m = np.asarray(self.mult); new_m = np.asarray(mult)
        old_b = np.asarray(self.blk, bool); new_b = np.asarray(blk, bool)
        changed = np.flatnonzero((old_m != new_m) | (old_b != new_b)).tolist()
        if not changed:
            return 0
        self._move_start(start)
        self.mult, self.blk = mult, blk
        gi = self.goal[0] * self.ny + self.goal[1]
        touched = set()
        for v in changed:
            for u, _ in self._nbrs(v):
                touched.add(u)
        for u in touched:
            if u != gi:
                self.rhs[u] = self._best_rhs(u)
            self._update_vertex(u, start)
        return len(changed)

    def _move_start(self, start):
        if self.last_start is not None and start != self.last_start:
            self.km += _octile(self.last_start[0], self.last_start[1], start[0], start[1])
        self.last_start = start

    def plan(self, start, max_expansions: int = 400_000) -> Optional[list]:
        """Cells from `start` to the goal, or None if the goal is unreachable."""
        start = (int(start[0]), int(start[1]))
        self._move_start(start)
        self._compute(start, max_expansions)
        si = start[0] * self.ny + start[1]
        if self.g[si] == INF and self.rhs[si] == INF:
            return None
        gi = self.goal[0] * self.ny + self.goal[1]
        path = [start]
        i, seen = si, {si}
        g = self.g
        while i != gi:
            best, best_j = INF, -1
            for j, L in self._nbrs(i):
                if self.blk[j]:
                    continue
                v = L * self.mult[j] + g[j]
                if v < best:
                    best, best_j = v, j
            if best_j < 0 or best == INF or best_j in seen:
                return None                    # inconsistent field; caller falls back
            seen.add(best_j)
            i = best_j
            path.append(divmod(i, self.ny))
        return path
