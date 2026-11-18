"""Dataset splitting.

A random split on ChEMBL bioactivity data leaks: kinase series are congeneric,
so near-identical analogs land on both sides and the model is scored on
molecules it has effectively already seen. Scaffold splitting puts whole
Bemis-Murcko series on one side, which is the regime the screener actually
operates in -- novel chemistry, not memorised analogs.

Both are provided so the gap between them can be reported rather than hidden.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def scaffold_groups(scaffolds) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, scaf in enumerate(scaffolds):
        groups[scaf].append(idx)
    return dict(groups)


def scaffold_split(scaffolds, test_frac=0.15, seed=0):
    """Capacity-based greedy scaffold assignment.

    Groups are walked largest-first and packed into train until it reaches its
    capacity; whatever no longer fits becomes test. Because assignment is driven
    by remaining capacity rather than by a per-group size rule, the split cannot
    invert when the data has only a few very large scaffolds -- a degenerate but
    entirely possible shape for a congeneric dataset.

    Largest-first also keeps the biggest series in train, where they carry the
    most signal, while test fills with the smaller and singleton scaffolds that
    better represent novel chemistry. Deterministic given a seed.
    """
    groups = scaffold_groups(scaffolds)
    # Sort by size desc, then by key, so ties are broken deterministically.
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    n_total = len(scaffolds)
    n_test_target = int(round(n_total * test_frac))
    train_capacity = n_total - n_test_target

    train_idx: list[int] = []
    test_idx: list[int] = []
    for _, members in order:
        if len(train_idx) + len(members) <= train_capacity:
            train_idx.extend(members)
        elif len(test_idx) + len(members) <= n_test_target:
            test_idx.extend(members)
        else:
            # Fits neither side. Train is the correct home: a scaffold larger
            # than the whole test quota would otherwise blow the test set past
            # the requested fraction -- in the extreme, inverting the split.
            # Some datasets simply cannot hit test_frac exactly; overshooting
            # train is the harmless direction to miss in.
            train_idx.extend(members)

    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def scaffold_kfold(scaffolds, n_splits=5, seed=0):
    """K-fold where every fold boundary respects scaffold membership.

    Used for the out-of-fold residuals that calibrate conformal intervals, so
    the calibration residuals reflect novel-scaffold error, not analog error.
    """
    groups = scaffold_groups(scaffolds)
    keys = sorted(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    # Greedy bin-packing: assign each scaffold to the currently smallest fold.
    fold_of_scaffold: dict[str, int] = {}
    fold_sizes = [0] * n_splits
    for key in sorted(keys, key=lambda k: -len(groups[k])):
        target = int(np.argmin(fold_sizes))
        fold_of_scaffold[key] = target
        fold_sizes[target] += len(groups[key])

    fold_ids = np.empty(len(scaffolds), dtype=int)
    for scaf, members in groups.items():
        fold_ids[members] = fold_of_scaffold[scaf]

    for k in range(n_splits):
        test_idx = np.where(fold_ids == k)[0]
        train_idx = np.where(fold_ids != k)[0]
        if len(test_idx) and len(train_idx):
            yield train_idx, test_idx


def random_kfold(n, n_splits=5, seed=0):
    """Plain random K-fold -- the optimistic baseline, reported for contrast."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_splits)
    for k in range(n_splits):
        test_idx = np.sort(folds[k])
        train_idx = np.sort(np.concatenate([folds[j] for j in range(n_splits) if j != k]))
        yield train_idx, test_idx
