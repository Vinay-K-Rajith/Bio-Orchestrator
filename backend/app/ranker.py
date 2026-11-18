"""Multi-objective lead ranking (Stage 3, "Model 2").

The plan specified a weighted sum of desirabilities *and* a Pareto front. Those
are two different algorithms answering two different questions, so both are
implemented and used for what each is actually good at:

  * Pareto (non-dominated sorting) answers "which molecules are not beaten on
    every axis at once?" It needs no weights, so it cannot be gamed by a bad
    weight choice -- but it does not produce a total order.
  * Desirability answers "given what the user asked for, which of those is
    best?" It produces the ranking, but only *within* a Pareto front, so a
    molecule that is dominated outright can never win on weights alone.

Ranking front-by-front is what makes the weights safe to expose to the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Objectives:
    """The competing goals, each normalised to [0, 1] where 1 is best."""

    affinity: float = 0.0
    confidence: float = 0.0
    safety: float = 0.0
    drug_likeness: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.affinity, self.confidence, self.safety,
                         self.drug_likeness], dtype=np.float64)


@dataclass
class Weights:
    """Relative importance, renormalised to sum to 1 on construction."""

    affinity: float = 0.40
    confidence: float = 0.20
    safety: float = 0.25
    drug_likeness: float = 0.15

    def normalised(self) -> np.ndarray:
        w = np.array([self.affinity, self.confidence, self.safety,
                      self.drug_likeness], dtype=np.float64)
        w = np.clip(w, 0.0, None)
        total = w.sum()
        return w / total if total > 0 else np.full(4, 0.25)


def desirability_affinity(pic50: float, floor: float = 5.0, ceil: float = 9.0) -> float:
    """Potency, linear in log space between a weak and an excellent binder.

    Log space is the right scale: the gap from pIC50 5 to 6 is a 10x potency
    improvement, the same as 8 to 9, so equal log steps deserve equal credit.
    """
    return float(np.clip((pic50 - floor) / (ceil - floor), 0.0, 1.0))


def desirability_confidence(interval_width: float, max_similarity: float,
                            in_domain: bool) -> float:
    """How much the affinity number can be trusted.

    Two orthogonal signals: how wide the calibrated interval is (the model's own
    admission of difficulty), and how close the molecule sits to known chemistry
    (whether the model has any right to an opinion). Both must be good.
    """
    # A 1-log-unit band is tight, a 4-log band is nearly useless.
    width_term = float(np.clip((4.0 - interval_width) / 3.0, 0.0, 1.0))
    sim_term = float(np.clip(max_similarity / 0.6, 0.0, 1.0))
    score = 0.5 * width_term + 0.5 * sim_term
    return score if in_domain else score * 0.5


def desirability_safety(n_alerts: int, herg_risk: float | None = None) -> float:
    """Structural alerts and, when the hERG model is available, cardiac risk.

    Alerts decay geometrically rather than hard-zeroing: one Brenk alert is a
    caution, four is a different category of problem.
    """
    alert_term = float(0.65 ** n_alerts)
    if herg_risk is None:
        return alert_term
    return alert_term * float(np.clip(1.0 - herg_risk, 0.0, 1.0))


def desirability_druglike(lipinski_violations: int, qed: float,
                          sa: float, veber_pass: bool) -> float:
    """Oral-lead plausibility: Lipinski, QED, synthesisability, Veber."""
    ro5 = {0: 1.0, 1: 0.7}.get(lipinski_violations, 0.2)
    qed_term = float(np.clip(qed, 0.0, 1.0))
    # SA runs 1 (trivial) to 10 (unmakeable); anything past ~6 is a red flag.
    sa_term = float(np.clip((6.0 - sa) / 3.0, 0.0, 1.0)) if np.isfinite(sa) else 0.5
    veber = 1.0 if veber_pass else 0.75
    return float(np.clip(0.45 * ro5 + 0.30 * qed_term + 0.25 * sa_term, 0, 1) * veber)


def non_dominated_sort(scores: np.ndarray) -> np.ndarray:
    """Assign each row a Pareto front index (0 = best front).

    Fast non-dominated sorting from NSGA-II (Deb et al., 2002). A dominates B if
    A is at least as good on every objective and strictly better on one.
    Vectorised per-individual, which is ample at screening-funnel sizes.
    """
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=int)

    fronts = np.full(n, -1, dtype=int)
    # domination_count[i] = how many solutions dominate i
    domination_count = np.zeros(n, dtype=int)
    dominated: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        ge = np.all(scores[i] >= scores, axis=1)
        gt = np.any(scores[i] > scores, axis=1)
        i_dominates = ge & gt          # i dominates these
        le = np.all(scores[i] <= scores, axis=1)
        lt = np.any(scores[i] < scores, axis=1)
        dominates_i = le & lt          # these dominate i
        dominated[i] = np.where(i_dominates)[0].tolist()
        domination_count[i] = int(dominates_i.sum())

    current = np.where(domination_count == 0)[0].tolist()
    front_id = 0
    while current:
        nxt: list[int] = []
        for i in current:
            fronts[i] = front_id
            for j in dominated[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    nxt.append(j)
        current = nxt
        front_id += 1

    fronts[fronts < 0] = front_id  # safety net for numerical ties
    return fronts


def crowding_distance(scores: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance -- rewards diversity within a front.

    Used as a tiebreak so the top-N are not four near-identical analogs.
    """
    n, m = scores.shape
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    for k in range(m):
        order = np.argsort(scores[:, k])
        col = scores[order, k]
        dist[order[0]] = dist[order[-1]] = np.inf
        span = col[-1] - col[0]
        if span > 0:
            dist[order[1:-1]] += (col[2:] - col[:-2]) / span
    return dist


@dataclass
class RankedCandidate:
    index: int
    objectives: Objectives
    utility: float
    pareto_front: int
    crowding: float
    rank: int = 0
    notes: list = field(default_factory=list)


def rank_candidates(objectives: list[Objectives], weights: Weights,
                    top_n: int = 5) -> list[RankedCandidate]:
    """Rank by Pareto front first, then weighted desirability within a front.

    Front-major ordering is the whole point: a molecule on front 0 with a
    mediocre utility still outranks a dominated molecule that scored well only
    because the weights happened to favour its one strong axis.
    """
    if not objectives:
        return []

    scores = np.vstack([o.as_array() for o in objectives])
    w = weights.normalised()
    utility = scores @ w
    fronts = non_dominated_sort(scores)

    crowd = np.zeros(len(scores))
    for f in np.unique(fronts):
        mask = fronts == f
        crowd[mask] = crowding_distance(scores[mask])

    order = sorted(
        range(len(scores)),
        key=lambda i: (fronts[i], -utility[i], -np.nan_to_num(crowd[i], posinf=1e9)),
    )

    ranked = []
    for rank, i in enumerate(order[:top_n], start=1):
        ranked.append(RankedCandidate(
            index=i, objectives=objectives[i], utility=float(utility[i]),
            pareto_front=int(fronts[i]),
            crowding=float(np.nan_to_num(crowd[i], posinf=999.0)),
            rank=rank,
        ))
    return ranked


def summarise_fronts(objectives: list[Objectives]) -> dict:
    if not objectives:
        return {"n": 0, "n_fronts": 0, "front_sizes": []}
    scores = np.vstack([o.as_array() for o in objectives])
    fronts = non_dominated_sort(scores)
    sizes = [int((fronts == f).sum()) for f in range(int(fronts.max()) + 1)]
    return {"n": len(objectives), "n_fronts": len(sizes), "front_sizes": sizes}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # A deliberate trap: one potent-but-unsafe molecule, one balanced molecule.
    objs = [
        Objectives(affinity=0.95, confidence=0.9, safety=0.10, drug_likeness=0.2),
        Objectives(affinity=0.70, confidence=0.8, safety=0.90, drug_likeness=0.85),
    ]
    objs += [Objectives(*rng.uniform(0.1, 0.8, 4)) for _ in range(30)]

    print("balanced weights:")
    for c in rank_candidates(objs, Weights(), top_n=3):
        o = c.objectives
        print(f"  #{c.rank} idx={c.index:<3} front={c.pareto_front} "
              f"util={c.utility:.3f}  aff={o.affinity:.2f} safe={o.safety:.2f}")

    print("\nsafety-prioritised weights:")
    for c in rank_candidates(objs, Weights(affinity=0.2, confidence=0.15,
                                           safety=0.5, drug_likeness=0.15), top_n=3):
        o = c.objectives
        print(f"  #{c.rank} idx={c.index:<3} front={c.pareto_front} "
              f"util={c.utility:.3f}  aff={o.affinity:.2f} safe={o.safety:.2f}")

    print("\nfronts:", summarise_fronts(objs))
