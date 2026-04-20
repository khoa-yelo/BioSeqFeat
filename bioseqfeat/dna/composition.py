"""
Composition-based DNA sequence feature extractors.

Classes
-------
MNC  -- mono-nucleotide composition  →  (4,)  frequency vector
DNC  -- di-nucleotide composition    → (16,)  pairwise frequency vector
TNC  -- tri-nucleotide composition   → (64,)  triplet frequency vector
"""

from __future__ import annotations

from itertools import product

import numpy as np

from ..base import Featurizer

# Canonical nucleotides in alphabetical order
NUCLEOTIDES: tuple[str, ...] = ("A", "C", "G", "T")
_NT_INDEX: dict[str, int] = {nt: i for i, nt in enumerate(NUCLEOTIDES)}
_NT_SET = frozenset(NUCLEOTIDES)


def _filter_seq(seq: str) -> str:
    """Upper-case *seq* and remove non-canonical nucleotides (A/C/G/T only)."""
    return "".join(c for c in seq.upper() if c in _NT_SET)


# ---------------------------------------------------------------------------
# MNC
# ---------------------------------------------------------------------------

class MNC(Featurizer):
    """Mono-nucleotide composition — 4-dimensional frequency vector.

    Elements are frequencies of A, C, G, T (alphabetical order).
    Non-canonical characters are silently ignored.

    Example
    -------
    >>> feat = MNC()
    >>> vec = feat.extract_one("AACGT")
    >>> vec.shape
    (4,)
    >>> round(float(vec.sum()), 6)
    1.0
    """

    name = "mnc"

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        seq = _filter_seq(seq)
        if not seq:
            return np.zeros(4, dtype=np.float32)
        counts = np.zeros(4, dtype=np.float32)
        for nt in seq:
            counts[_NT_INDEX[nt]] += 1.0
        counts /= len(seq)
        return counts


# ---------------------------------------------------------------------------
# DNC
# ---------------------------------------------------------------------------

class DNC(Featurizer):
    """Di-nucleotide composition — 16-dimensional pairwise frequency vector.

    Each of the 16 elements is the fraction of the corresponding ordered
    nucleotide pair (i, j) among all consecutive dinucleotides.
    Positions are ordered lexicographically: AA, AC, AG, AT, CA, …, TT.

    Example
    -------
    >>> feat = DNC()
    >>> vec = feat.extract_one("AACGT")
    >>> vec.shape
    (16,)
    >>> round(float(vec.sum()), 6)
    1.0
    """

    name = "dnc"

    _DI_INDEX: dict[tuple[str, str], int] = {
        (a, b): i * 4 + j
        for i, a in enumerate(NUCLEOTIDES)
        for j, b in enumerate(NUCLEOTIDES)
    }

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        seq = _filter_seq(seq)
        counts = np.zeros(16, dtype=np.float32)
        n_di = len(seq) - 1
        if n_di <= 0:
            return counts
        for k in range(n_di):
            counts[self._DI_INDEX[(seq[k], seq[k + 1])]] += 1.0
        counts /= n_di
        return counts


# ---------------------------------------------------------------------------
# TNC
# ---------------------------------------------------------------------------

class TNC(Featurizer):
    """Tri-nucleotide composition — 64-dimensional triplet frequency vector.

    Captures codon-level usage patterns.  Each element is the fraction of
    the corresponding ordered triplet (i, j, k) among all consecutive
    trinucleotides.  Positions are ordered lexicographically: AAA, AAC, …, TTT.

    Example
    -------
    >>> feat = TNC()
    >>> vec = feat.extract_one("AACGT")
    >>> vec.shape
    (64,)
    >>> round(float(vec.sum()), 6)
    1.0
    """

    name = "tnc"

    _TRI_INDEX: dict[tuple[str, str, str], int] = {
        (a, b, c): i * 16 + j * 4 + k
        for i, a in enumerate(NUCLEOTIDES)
        for j, b in enumerate(NUCLEOTIDES)
        for k, c in enumerate(NUCLEOTIDES)
    }

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        seq = _filter_seq(seq)
        counts = np.zeros(64, dtype=np.float32)
        n_tri = len(seq) - 2
        if n_tri <= 0:
            return counts
        for k in range(n_tri):
            counts[self._TRI_INDEX[(seq[k], seq[k + 1], seq[k + 2])]] += 1.0
        counts /= n_tri
        return counts
