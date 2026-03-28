"""
Global physicochemical descriptors for protein sequences.

Classes
-------
GlobalDescriptors -- 9 scalar descriptors → (9,) vector
"""

from __future__ import annotations

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from ..base import Featurizer

# Canonical 20 amino acids used to strip non-standard residues
_CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _filter_seq(seq: str) -> str:
    """Return *seq* upper-cased with non-canonical residues removed."""
    return "".join(r for r in seq.upper() if r in _CANONICAL)


def _aliphatic_index(seq: str) -> float:
    """Aliphatic index: relative volume occupied by aliphatic side chains.

    Formula (Ikai 1980)::

        AI = (A + 2.9*V + 3.9*(I + L)) / L * 100

    where each term is the count of the respective amino acid and *L* is the
    sequence length.
    """
    L = len(seq)
    if L == 0:
        return 0.0
    A = seq.count("A")
    V = seq.count("V")
    I = seq.count("I")
    leu = seq.count("L")
    return (A + 2.9 * V + 3.9 * (I + leu)) / L * 100.0


class GlobalDescriptors(Featurizer):
    """Nine global physicochemical descriptors — (9,) vector.

    Uses ``Bio.SeqUtils.ProtParam.ProteinAnalysis`` for most properties, plus
    a local implementation of the aliphatic index.  Non-canonical residues are
    silently removed before analysis.

    Output dimensions (in order)
    -----------------------------
    0  gravy        -- GRAVY index (mean hydrophobicity, Kyte–Doolittle)
    1  pi           -- isoelectric point
    2  aromaticity  -- fraction of F + W + Y residues
    3  instability  -- instability index (Guruprasad 1990)
    4  aliphatic    -- aliphatic index × 100 (Ikai 1980)
    5  mw           -- molecular weight (Da)
    6  helix        -- predicted helix fraction
    7  turn         -- predicted turn fraction
    8  sheet        -- predicted sheet fraction

    Returns a zero vector for sequences that become empty after filtering.

    Example
    -------
    >>> feat = GlobalDescriptors()
    >>> vec = feat.extract_one("ACDEFGHIKLMNPQRSTVWY")
    >>> vec.shape
    (9,)
    """

    name = "global_descriptors"

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        seq = _filter_seq(seq)
        if not seq:
            return np.zeros(9, dtype=np.float32)

        pa = ProteinAnalysis(seq)
        helix, turn, sheet = pa.secondary_structure_fraction()

        values = [
            pa.gravy(),
            pa.isoelectric_point(),
            pa.aromaticity(),
            pa.instability_index(),
            _aliphatic_index(seq),
            pa.molecular_weight(),
            helix,
            turn,
            sheet,
        ]
        return np.array(values, dtype=np.float32)
