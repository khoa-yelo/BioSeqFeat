"""
Global physicochemical descriptors for DNA sequences.

Classes
-------
GlobalDescriptors -- 12 scalar descriptors → (12,) vector
"""

from __future__ import annotations

import math

import numpy as np

from ..base import Featurizer

_NT_SET = frozenset("ACGT")


def _filter_seq(seq: str) -> str:
    return "".join(c for c in seq.upper() if c in _NT_SET)


class GlobalDescriptors(Featurizer):
    """Twelve global physicochemical descriptors for DNA — (12,) vector.

    All descriptors are length-invariant (fractions, ratios, log-scale length)
    except where noted.

    Output dimensions (in order)
    -----------------------------
    0  gc_content      -- fraction of G + C
    1  at_content      -- fraction of A + T  (= 1 − gc_content)
    2  gc_skew         -- (G − C) / (G + C); 0.0 if G + C = 0
    3  at_skew         -- (A − T) / (A + T); 0.0 if A + T = 0
    4  cpg_oe          -- CpG observed/expected = f(CG) / (f(C)·f(G));
                          expected to be <1 in CpG-depleted regions,
                          ~1 in CpG islands; 0 if denominator is 0
    5  purine_frac     -- fraction of A + G
    6  keto_frac       -- fraction of G + T  (Keto group: G and T)
    7  tm_estimate     -- melting temperature estimate (°C):
                          2·(A+T) + 4·(G+C) (Wallace rule)
    8  shannon_entropy -- per-nucleotide Shannon entropy (bits, base 2)
    9  complexity      -- linguistic complexity = observed k-mers / max k-mers
                          (k=3 trinucleotides; proxy for sequence repetitiveness)
    10 pupy_ratio      -- purine / pyrimidine ratio; 0 if no pyrimidines
    11 log_length      -- log2(len + 1), captures rough sequence length scale

    Returns a zero vector for sequences that are empty after filtering.

    Example
    -------
    >>> feat = GlobalDescriptors()
    >>> vec = feat.extract_one("ATGCATGCATGC")
    >>> vec.shape
    (12,)
    """

    name = "dna_global_descriptors"

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        seq = _filter_seq(seq)
        L = len(seq)
        if L == 0:
            return np.zeros(12, dtype=np.float32)

        a = seq.count("A")
        c = seq.count("C")
        g = seq.count("G")
        t = seq.count("T")

        gc = g + c
        at = a + t

        gc_content = gc / L
        at_content = at / L

        gc_skew = (g - c) / gc if gc > 0 else 0.0
        at_skew = (a - t) / at if at > 0 else 0.0

        # CpG O/E: count overlapping CG dinucleotides
        cpg_obs = sum(1 for k in range(L - 1) if seq[k] == "C" and seq[k + 1] == "G")
        cpg_exp = (c / L) * (g / L) if (c > 0 and g > 0) else 0.0
        cpg_oe = (cpg_obs / (L - 1)) / cpg_exp if (cpg_exp > 0 and L > 1) else 0.0

        purine_frac = (a + g) / L
        keto_frac = (g + t) / L

        # Wallace rule melting temperature (useful up to ~14 bp; kept as rough proxy)
        tm_estimate = 2.0 * at + 4.0 * gc

        # Shannon entropy over the 4 nucleotide frequencies
        freqs = np.array([a, c, g, t], dtype=np.float64) / L
        entropy = float(-sum(f * math.log2(f) for f in freqs if f > 0))

        # Linguistic complexity: fraction of distinct trinucleotides observed
        if L >= 3:
            tri_set = set(seq[k:k + 3] for k in range(L - 2))
            max_tris = min(64, L - 2)
            complexity = len(tri_set) / max_tris
        else:
            complexity = 0.0

        pupy_ratio = (a + g) / (c + t) if (c + t) > 0 else 0.0
        log_length = math.log2(L + 1)

        values = [
            gc_content,
            at_content,
            gc_skew,
            at_skew,
            cpg_oe,
            purine_frac,
            keto_frac,
            tm_estimate,
            entropy,
            complexity,
            pupy_ratio,
            log_length,
        ]
        return np.array(values, dtype=np.float32)
