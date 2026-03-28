"""
CTD (Composition / Transition / Distribution) protein sequence features.

Classes
-------
CTD -- 147-dimensional CTD feature vector

Background
----------
Dubchak et al. (1995) introduced CTD descriptors to encode both the amino-acid
class composition and the sequential topology of a protein.  Seven
physicochemical properties partition the 20 canonical amino acids into three
classes each.  For every property the descriptor captures:

  * Composition  (C) — fraction of residues in each class          →  3 values
  * Transition   (T) — transition frequency between class pairs     →  3 values
  * Distribution (D) — sequence position of the 0/25/50/75/100th
                        percentile residue for each class           → 15 values

Total per property: 3 + 3 + 15 = 21.  Over 7 properties: **147 dimensions**.

Output vector layout
--------------------
  [  0: 21] Composition  — (prop0_c0, prop0_c1, prop0_c2, prop1_c0, …, prop6_c2)
  [ 21: 42] Transition   — (prop0_t01, prop0_t02, prop0_t12, …,     prop6_t12)
  [ 42:147] Distribution — (prop0_c0_p0, …, prop0_c0_p100, prop0_c1_p0, …)

References
----------
Dubchak, I., Muchnik, I., Holbrook, S. R., & Kim, S. H. (1995).
  Prediction of protein folding class using global description of amino acid
  sequence. *PNAS*, 92(19), 8700–8704.
"""

from __future__ import annotations

import numpy as np

from ..base import Featurizer

# ---------------------------------------------------------------------------
# Property definitions (Dubchak 1995)
# Each entry: (property_name, [class0_residues, class1_residues, class2_residues])
# Every one of the 20 canonical AAs appears in exactly one class per property.
# ---------------------------------------------------------------------------
_PROPERTY_DEFS: list[tuple[str, list[str]]] = [
    (
        "hydrophobicity",
        ["RKDEQN",           # polar
         "GASTPHY",          # neutral  (note: Y here per Dubchak; H debated)
         "CLVIMFW"],         # hydrophobic
    ),
    (
        "vdw_volume",
        ["GASTPDC",          # small   (0 – 2.78)
         "NVEQIL",           # medium  (2.95 – 4.0)
         "MHKFRYW"],         # large   (4.43 – 8.08)
    ),
    (
        "polarity",
        ["LIFWCMVY",         # low polarity
         "PATGS",            # medium
         "HQRKNED"],         # high polarity
    ),
    (
        "polarizability",
        ["GASDT",            # low   (0 – 0.108)
         "CPNVEQIL",         # medium (0.128 – 0.186)
         "KMHFRYW"],         # high  (0.219 – 0.409)
    ),
    (
        "charge",
        ["KR",               # positive
         "ANCQGHILMFPSTWYV", # neutral
         "DE"],              # negative
    ),
    (
        "secondary_structure",
        ["EALMQKRH",         # helix
         "VIYCWFT",          # strand
         "GNPSD"],           # coil
    ),
    (
        "solvent_accessibility",
        ["ALFCGIVW",         # buried
         "RHQTSYMP",         # intermediate
         "KNED"],            # exposed
    ),
]

# Validate at import time: every property must cover all 20 canonical AAs exactly once.
_CANONICAL_SET = frozenset("ACDEFGHIKLMNPQRSTVWY")

def _build_lookups() -> list[dict[str, int]]:
    lookups = []
    for name, groups in _PROPERTY_DEFS:
        lookup: dict[str, int] = {}
        covered: set[str] = set()
        for cls_idx, residues in enumerate(groups):
            for aa in residues:
                if aa not in _CANONICAL_SET:
                    raise ValueError(
                        f"Property '{name}': '{aa}' is not a canonical amino acid."
                    )
                if aa in covered:
                    raise ValueError(
                        f"Property '{name}': '{aa}' appears in more than one class."
                    )
                lookup[aa] = cls_idx
                covered.add(aa)
        missing = _CANONICAL_SET - covered
        if missing:
            raise ValueError(
                f"Property '{name}': residues {sorted(missing)} not assigned to any class."
            )
        lookups.append(lookup)
    return lookups


_LOOKUPS: list[dict[str, int]] = _build_lookups()
_N_PROPS = len(_LOOKUPS)  # 7
_N_DIM = _N_PROPS * (3 + 3 + 15)  # 147

# Transition pairs between the 3 classes: (c_low, c_high) ordered
_TRANS_PAIRS = [(0, 1), (0, 2), (1, 2)]  # 3 pairs

# Distribution percentile fractions
_PCT_FRACS = [0.0, 0.25, 0.50, 0.75, 1.0]


def _filter_seq(seq: str) -> str:
    return "".join(r for r in seq.upper() if r in _CANONICAL_SET)


# ---------------------------------------------------------------------------
# CTD featurizer
# ---------------------------------------------------------------------------

class CTD(Featurizer):
    """Composition / Transition / Distribution protein descriptors — (147,) vector.

    Encodes both local amino-acid class fractions and the sequential topology
    of the sequence using seven physicochemical property groupings.

    Output dimensions
    -----------------
    147 = 7 properties × (3 composition + 3 transition + 15 distribution)

    Vector layout::

        [  0: 21]  Composition   — prop × class
        [ 21: 42]  Transition    — prop × class-pair  (pairs: 0↔1, 0↔2, 1↔2)
        [ 42:147]  Distribution  — prop × class × percentile (0,25,50,75,100%)

    All position-based values are expressed as fractions of the sequence length
    so that the descriptor is length-invariant.  Returns a zero vector for
    sequences that are empty after filtering non-canonical residues.

    Properties and class groupings
    --------------------------------
    1. hydrophobicity      : polar | neutral | hydrophobic
    2. vdw_volume          : small | medium  | large
    3. polarity            : low   | medium  | high
    4. polarizability      : low   | medium  | high
    5. charge              : positive | neutral | negative
    6. secondary_structure : helix | strand | coil
    7. solvent_accessibility: buried | intermediate | exposed

    References
    ----------
    Dubchak et al. (1995) PNAS 92(19):8700–8704.

    Example
    -------
    >>> feat = CTD()
    >>> vec = feat.extract_one("ACDEFGHIKLMNPQRSTVWY")
    >>> vec.shape
    (147,)
    """

    name = "ctd"

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        seq = _filter_seq(seq)
        L = len(seq)
        if L == 0:
            return np.zeros(_N_DIM, dtype=np.float32)

        comp = np.zeros(_N_PROPS * 3, dtype=np.float32)        # [0:21]
        trans = np.zeros(_N_PROPS * 3, dtype=np.float32)       # [21:42]
        dist = np.zeros(_N_PROPS * 3 * 5, dtype=np.float32)    # [42:147]

        for pi, lookup in enumerate(_LOOKUPS):
            labels = np.fromiter(
                (lookup[aa] for aa in seq), dtype=np.int8, count=L
            )

            # --- Composition ---
            c_off = pi * 3
            for c in range(3):
                comp[c_off + c] = np.sum(labels == c) / L

            # --- Transition ---
            t_off = pi * 3
            if L > 1:
                a, b = labels[:-1], labels[1:]
                for ti, (c1, c2) in enumerate(_TRANS_PAIRS):
                    count = int(
                        np.sum(((a == c1) & (b == c2)) | ((a == c2) & (b == c1)))
                    )
                    trans[t_off + ti] = count / (L - 1)

            # --- Distribution ---
            d_off = pi * 15
            for c in range(3):
                positions = np.where(labels == c)[0]  # 0-indexed
                n_c = len(positions)
                slot = d_off + c * 5
                if n_c == 0:
                    # dist[slot:slot+5] already 0
                    continue
                for fi, frac in enumerate(_PCT_FRACS):
                    if frac == 0.0:
                        idx = 0
                    else:
                        idx = int(np.ceil(frac * n_c)) - 1
                    dist[slot + fi] = (positions[idx] + 1) / L  # 1-indexed / L

        return np.concatenate([comp, trans, dist])

    @staticmethod
    def feature_names() -> list[str]:
        """Return the 147 feature names in output order."""
        prop_names = [p for p, _ in _PROPERTY_DEFS]
        names: list[str] = []
        for pn in prop_names:
            for c in range(3):
                names.append(f"C_{pn}_c{c}")
        for pn in prop_names:
            for c1, c2 in _TRANS_PAIRS:
                names.append(f"T_{pn}_c{c1}c{c2}")
        for pn in prop_names:
            for c in range(3):
                for pct in [0, 25, 50, 75, 100]:
                    names.append(f"D_{pn}_c{c}_p{pct}")
        return names
