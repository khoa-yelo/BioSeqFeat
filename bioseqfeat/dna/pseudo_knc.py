"""
Pseudo K-mer Nucleotide Composition (PseKNC) for DNA sequences.

Classes
-------
PseKNC -- (16 + λ)-dimensional pseudo-dinucleotide composition vector

Background
----------
Analogous to PseAAC (Chou 2001) but for DNA.  The base 16-d dinucleotide
composition is augmented with λ sequence-order correlation factors derived
from physicochemical properties of consecutive dinucleotides.

The correlation factor of rank k::

    θ_k = 1/(L-k-1) * Σ_{i=1}^{L-k-1} Θ(dn_i, dn_{i+k})

where dn_i is the dinucleotide starting at position i, and Θ is the average
squared difference in normalised physicochemical properties.

Five dinucleotide physicochemical properties are used (Breslauer 1986;
Sugimoto 1996; Ulanovsky 1986):
  1. Stacking energy    (ΔG, kcal/mol)
  2. Enthalpy           (ΔH, kcal/mol)
  3. Entropy            (ΔS, cal/mol/K)
  4. Shift              (slide parameter, Å)
  5. Slide              (shift parameter, Å)

References
----------
Chen et al. (2014). PseKNC: A flexible web server for generating
pseudo K-tuple nucleotide composition. *Anal. Biochem.*, 456, 53–60.
"""

from __future__ import annotations

import numpy as np

from ..base import Featurizer
from .composition import NUCLEOTIDES, _filter_seq, DNC

_NT_SET = frozenset(NUCLEOTIDES)

# ---------------------------------------------------------------------------
# Dinucleotide physicochemical properties (16 dinucleotides, alphabetical)
# Order: AA, AC, AG, AT, CA, CC, CG, CT, GA, GC, GG, GT, TA, TC, TG, TT
# Sources: Breslauer (1986), Sugimoto (1996), Ulanovsky (1986)
# ---------------------------------------------------------------------------
_DINUCS = [
    a + b
    for a in NUCLEOTIDES
    for b in NUCLEOTIDES
]  # 16 entries, same order as DNC

# Each column: [stacking_energy, enthalpy, entropy, shift, slide]
# Values from Chen et al. (2014) supplementary / iLearn database
_PROPS_RAW: dict[str, tuple[float, float, float, float, float]] = {
    #      ΔG       ΔH       ΔS    shift   slide
    "AA": (-1.02,  -7.9,  -22.2,  0.06,  0.5),
    "AC": (-1.01,  -7.1,  -19.8,  0.10,  0.2),
    "AG": (-0.98,  -7.8,  -21.0,  0.02, -0.6),
    "AT": (-0.88,  -7.2,  -20.4,  0.03, -0.1),
    "CA": (-0.90,  -8.5,  -22.7,  0.18, -0.8),
    "CC": (-1.30,  -8.0,  -19.9, -0.04, -0.2),
    "CG": (-2.17, -10.6,  -27.2, -0.06, -0.7),
    "CT": (-0.98,  -7.8,  -21.0,  0.02, -0.6),
    "GA": (-1.11,  -8.2,  -22.2,  0.04, -0.6),
    "GC": (-2.24, -10.4,  -26.4,  0.01, -0.3),
    "GG": (-1.30,  -8.0,  -19.9, -0.04, -0.2),
    "GT": (-1.01,  -7.1,  -19.8,  0.10,  0.2),
    "TA": (-0.61,  -7.2,  -21.3,  0.23,  1.0),
    "TC": (-0.98,  -7.8,  -21.0,  0.02, -0.6),
    "TG": (-0.90,  -8.5,  -22.7,  0.18, -0.8),
    "TT": (-1.02,  -7.9,  -22.2,  0.06,  0.5),
}

_N_PROPS = 5
_DI_ORDER = _DINUCS  # consistent ordering

# Build array: shape (N_PROPS, 16)
_RAW_ARR = np.array(
    [[_PROPS_RAW[dn][p] for dn in _DI_ORDER] for p in range(_N_PROPS)],
    dtype=np.float64,
)
# Normalize each property to zero mean, unit std
_mean = _RAW_ARR.mean(axis=1, keepdims=True)
_std  = _RAW_ARR.std(axis=1, keepdims=True, ddof=0)
_NORM_PROPS: np.ndarray = (_RAW_ARR - _mean) / (_std + 1e-12)  # (5, 16)

# Index: dinucleotide string -> column index in _NORM_PROPS
_DI_IDX: dict[str, int] = {dn: i for i, dn in enumerate(_DI_ORDER)}


class PseKNC(Featurizer):
    """Pseudo K-mer Nucleotide Composition — (16 + λ)-dimensional vector.

    Extends the standard 16-d dinucleotide composition with λ sequence-order
    correlation factors capturing long-range physicochemical interactions.

    Parameters
    ----------
    lambda_ : int
        Number of pseudo correlation components. Must be < sequence length − 1.
        Defaults to 20.
    weight : float
        Relative weight w for the pseudo components. Defaults to 0.05.

    Example
    -------
    >>> feat = PseKNC(lambda_=10)
    >>> vec = feat.extract_one("ATGCATGCATGCATGC")
    >>> vec.shape
    (26,)
    >>> round(float(vec.sum()), 6)
    1.0
    """

    name = "pseknc"

    def __init__(self, lambda_: int = 20, weight: float = 0.05):
        if lambda_ < 1:
            raise ValueError("lambda_ must be >= 1.")
        if weight < 0:
            raise ValueError("weight must be non-negative.")
        self.lambda_ = lambda_
        self.weight = weight

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        seq = _filter_seq(seq)
        L = len(seq)
        dim_total = 16 + self.lambda_

        # Dinucleotides that can be extracted
        n_di = L - 1
        if n_di <= 0:
            return np.zeros(dim_total, dtype=np.float32)

        lam = min(self.lambda_, n_di - 1)  # need at least lam+1 dinucleotides

        # --- Dinucleotide frequencies (base 16-d) ---
        f = np.zeros(16, dtype=np.float64)
        for k in range(n_di):
            di = seq[k:k + 2]
            f[_DI_IDX[di]] += 1.0
        f /= n_di

        if lam == 0:
            vec = np.zeros(dim_total, dtype=np.float32)
            vec[:16] = f.astype(np.float32)
            return vec

        # --- Physicochemical property values at each dinucleotide position ---
        # H[k] = (5,) vector of normalised properties for dinucleotide at pos k
        H = np.array([_NORM_PROPS[:, _DI_IDX[seq[k:k + 2]]]
                      for k in range(n_di)])  # (n_di, 5)

        # --- Correlation factors θ_k ---
        theta = np.zeros(lam, dtype=np.float64)
        for k in range(1, lam + 1):
            diff = H[:n_di - k] - H[k:]          # (n_di-k, 5)
            theta[k - 1] = np.mean(np.mean(diff ** 2, axis=1))

        # --- Normalised PseKNC vector ---
        denom = 1.0 + self.weight * theta.sum()
        vec = np.zeros(dim_total, dtype=np.float32)
        vec[:16] = (f / denom).astype(np.float32)
        vec[16:16 + lam] = (self.weight * theta / denom).astype(np.float32)
        # Remaining positions (if lam < lambda_) stay 0 — short sequence padding

        return vec
