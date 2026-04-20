"""
CTD (Composition / Transition / Distribution) DNA sequence features.

Classes
-------
CTD -- 39-dimensional CTD feature vector

Background
----------
Adapted from Dubchak et al. (1995) for DNA sequences.  Three physicochemical
properties partition the 4 canonical nucleotides into two groups each.  For
every property the descriptor captures:

  * Composition  (C) — fraction of nucleotides in each group        →  2 values
  * Transition   (T) — fraction of adjacent pairs crossing groups   →  1 value
  * Distribution (D) — sequence positions of the 0/25/50/75/100th
                        percentile nucleotide for each group         → 10 values

Total per property: 2 + 1 + 10 = 13.  Over 3 properties: **39 dimensions**.

Properties
----------
1. Ring structure (purine / pyrimidine)
   Purine    : A, G  (double-ring)
   Pyrimidine: C, T  (single-ring)

2. Functional group (amino / keto)
   Amino: A, C  (–NH₂ group)
   Keto : G, T  (C=O group)

3. Hydrogen-bond strength (strong / weak)
   Strong (3 H-bonds): G, C
   Weak   (2 H-bonds): A, T

Output vector layout
--------------------
  [  0:  6]  Composition   — (prop0_g0, prop0_g1, prop1_g0, …, prop2_g1)
  [  6:  9]  Transition    — (prop0, prop1, prop2)
  [  9: 39]  Distribution  — (prop0_g0_p0..p100, prop0_g1_p0..p100, …)
"""

from __future__ import annotations

import numpy as np

from ..base import Featurizer

_NT_SET = frozenset("ACGT")

# Each entry: (property_name, [group0_nucleotides, group1_nucleotides])
_PROPERTY_DEFS: list[tuple[str, list[str]]] = [
    ("ring_structure",   ["AG",  "CT"]),   # purine | pyrimidine
    ("functional_group", ["AC",  "GT"]),   # amino  | keto
    ("h_bond_strength",  ["GC",  "AT"]),   # strong | weak
]

# Build lookup: property_index -> {nucleotide: group_index (0 or 1)}
def _build_lookups() -> list[dict[str, int]]:
    lookups = []
    for name, groups in _PROPERTY_DEFS:
        lookup: dict[str, int] = {}
        covered: set[str] = set()
        for gi, nucleotides in enumerate(groups):
            for nt in nucleotides:
                if nt not in _NT_SET:
                    raise ValueError(f"Property '{name}': '{nt}' is not canonical.")
                if nt in covered:
                    raise ValueError(f"Property '{name}': '{nt}' in more than one group.")
                lookup[nt] = gi
                covered.add(nt)
        missing = _NT_SET - covered
        if missing:
            raise ValueError(f"Property '{name}': {sorted(missing)} not assigned.")
        lookups.append(lookup)
    return lookups


_LOOKUPS: list[dict[str, int]] = _build_lookups()
_N_PROPS = len(_LOOKUPS)     # 3
_N_GROUPS = 2                 # binary partition
_N_DIM = _N_PROPS * (2 + 1 + 10)  # 39

_PCT_FRACS = [0.0, 0.25, 0.50, 0.75, 1.0]


def _filter_seq(seq: str) -> str:
    return "".join(c for c in seq.upper() if c in _NT_SET)


class CTD(Featurizer):
    """Composition / Transition / Distribution DNA descriptors — (39,) vector.

    Encodes both local nucleotide group fractions and sequential topology
    using three physicochemical property groupings.

    Output dimensions
    -----------------
    39 = 3 properties × (2 composition + 1 transition + 10 distribution)

    Vector layout::

        [  0:  6]  Composition   — prop × group
        [  6:  9]  Transition    — prop (one value: fraction of transitions)
        [  9: 39]  Distribution  — prop × group × percentile (0,25,50,75,100%)

    Properties and groupings
    -------------------------
    1. ring_structure   : purine (A,G) | pyrimidine (C,T)
    2. functional_group : amino (A,C)  | keto (G,T)
    3. h_bond_strength  : strong (G,C) | weak (A,T)

    References
    ----------
    Adapted from Dubchak et al. (1995) PNAS 92(19):8700–8704.

    Example
    -------
    >>> feat = CTD()
    >>> vec = feat.extract_one("ATGCATGCATGC")
    >>> vec.shape
    (39,)
    """

    name = "dna_ctd"

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        seq = _filter_seq(seq)
        L = len(seq)
        if L == 0:
            return np.zeros(_N_DIM, dtype=np.float32)

        comp  = np.zeros(_N_PROPS * _N_GROUPS, dtype=np.float32)   # [0:6]
        trans = np.zeros(_N_PROPS, dtype=np.float32)                # [6:9]
        dist  = np.zeros(_N_PROPS * _N_GROUPS * 5, dtype=np.float32)  # [9:39]

        for pi, lookup in enumerate(_LOOKUPS):
            labels = np.fromiter(
                (lookup[nt] for nt in seq), dtype=np.int8, count=L
            )

            # --- Composition ---
            c_off = pi * _N_GROUPS
            for g in range(_N_GROUPS):
                comp[c_off + g] = np.sum(labels == g) / L

            # --- Transition (fraction of adjacent pairs from different groups) ---
            if L > 1:
                a, b = labels[:-1], labels[1:]
                trans[pi] = float(np.sum(a != b)) / (L - 1)

            # --- Distribution ---
            d_off = pi * (_N_GROUPS * 5)
            for g in range(_N_GROUPS):
                positions = np.where(labels == g)[0]  # 0-indexed
                n_g = len(positions)
                slot = d_off + g * 5
                if n_g == 0:
                    continue
                for fi, frac in enumerate(_PCT_FRACS):
                    if frac == 0.0:
                        idx = 0
                    else:
                        idx = int(np.ceil(frac * n_g)) - 1
                    dist[slot + fi] = (positions[idx] + 1) / L  # 1-indexed / L

        return np.concatenate([comp, trans, dist])

    @staticmethod
    def feature_names() -> list[str]:
        """Return the 39 feature names in output order."""
        prop_names = [p for p, _ in _PROPERTY_DEFS]
        group_names = [["purine", "pyrimidine"], ["amino", "keto"], ["strong", "weak"]]
        names: list[str] = []
        for pi, pn in enumerate(prop_names):
            for g in range(_N_GROUPS):
                names.append(f"C_{pn}_g{g}")
        for pn in prop_names:
            names.append(f"T_{pn}")
        for pi, pn in enumerate(prop_names):
            for g in range(_N_GROUPS):
                for pct in [0, 25, 50, 75, 100]:
                    names.append(f"D_{pn}_g{g}_p{pct}")
        return names
