"""
Composition-based protein sequence feature extractors.

Classes
-------
AAC      -- amino acid composition → (20,) frequency vector
DPC      -- dipeptide composition  → (400,) pairwise frequency vector
PseAAC   -- pseudo amino acid composition (Chou 2001) → (20 + lambda_,) vector
"""

from __future__ import annotations

from itertools import product

import numpy as np

from ..base import Featurizer

# Canonical ordering used throughout (alphabetical by one-letter code)
AMINO_ACIDS: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")
_AA_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# ---------------------------------------------------------------------------
# Physicochemical properties for PseAAC (Chou 2001)
# Values: hydrophobicity (Tanford 1962), hydrophilicity (Hopp-Woods 1981),
#         side-chain mass.
# ---------------------------------------------------------------------------
_PROPS: dict[str, tuple[float, float, float]] = {
    #       H1      H2      mass
    "A": ( 0.62,  -0.50,   15.0),
    "C": ( 0.29,  -1.00,   47.0),
    "D": (-0.90,   3.00,   59.0),
    "E": (-0.74,   3.00,   73.0),
    "F": ( 1.19,  -2.50,   91.0),
    "G": ( 0.48,   0.00,    1.0),
    "H": (-0.40,  -0.50,   82.0),
    "I": ( 1.38,  -1.80,   57.0),
    "K": (-1.50,   3.00,   73.0),
    "L": ( 1.06,  -1.80,   57.0),
    "M": ( 0.64,  -1.30,   75.0),
    "N": (-0.78,   0.20,   58.0),
    "P": ( 0.12,   0.00,   42.0),
    "Q": (-0.85,   0.20,   72.0),
    "R": (-2.53,   3.00,  101.0),
    "S": (-0.18,   0.30,   31.0),
    "T": (-0.05,  -0.40,   45.0),
    "V": ( 1.08,  -1.50,   43.0),
    "W": ( 0.81,  -3.40,  130.0),
    "Y": ( 0.26,  -2.30,  107.0),
}
_N_PROPS = 3

# Normalise each property to zero mean, unit std across the 20 canonical AAs
_RAW = np.array([[_PROPS[aa][p] for aa in AMINO_ACIDS] for p in range(_N_PROPS)],
                dtype=np.float64)  # shape (3, 20)
_mean = _RAW.mean(axis=1, keepdims=True)
_std  = _RAW.std(axis=1, keepdims=True, ddof=0)
_NORM_PROPS: np.ndarray = (_RAW - _mean) / _std   # shape (3, 20)


def _filter_seq(seq: str) -> str:
    """Return *seq* with non-canonical residues removed (silent)."""
    return "".join(r for r in seq.upper() if r in _AA_INDEX)


# ---------------------------------------------------------------------------
# AAC
# ---------------------------------------------------------------------------

class AAC(Featurizer):
    """Amino acid composition — 20-dimensional frequency vector.

    Each element is the fraction of the corresponding amino acid in the
    sequence.  The 20 positions follow alphabetical one-letter code order:
    ``A C D E F G H I K L M N P Q R S T V W Y``.

    Non-canonical residues are silently ignored.

    Example
    -------
    >>> feat = AAC()
    >>> vec = feat.extract_one("ACDEFGHIKLMNPQRSTVWY")
    >>> vec.shape
    (20,)
    >>> vec.sum()
    np.float64(1.0)
    """

    name = "aac"

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        seq = _filter_seq(seq)
        if not seq:
            return np.zeros(20, dtype=np.float32)
        counts = np.zeros(20, dtype=np.float32)
        for aa in seq:
            counts[_AA_INDEX[aa]] += 1.0
        counts /= len(seq)
        return counts


# ---------------------------------------------------------------------------
# DPC
# ---------------------------------------------------------------------------

class DPC(Featurizer):
    """Dipeptide composition — 400-dimensional pairwise frequency vector.

    Each of the 400 elements is the fraction of the corresponding ordered
    amino-acid pair (i, j) among all consecutive dipeptides in the sequence.
    Positions are ordered lexicographically: AA, AC, AD, …, YY.

    Non-canonical residues are silently ignored (the enclosing dipeptide is
    dropped).

    Example
    -------
    >>> feat = DPC()
    >>> vec = feat.extract_one("ACDEFGHIKLMNPQRSTVWY")
    >>> vec.shape
    (400,)
    >>> round(float(vec.sum()), 6)
    1.0
    """

    name = "dpc"

    # Build a static index: (aa1, aa2) -> flat position
    _DI_INDEX: dict[tuple[str, str], int] = {
        (a, b): i * 20 + j
        for i, a in enumerate(AMINO_ACIDS)
        for j, b in enumerate(AMINO_ACIDS)
    }

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        seq = _filter_seq(seq)
        counts = np.zeros(400, dtype=np.float32)
        n_di = len(seq) - 1
        if n_di <= 0:
            return counts
        for k in range(n_di):
            counts[self._DI_INDEX[(seq[k], seq[k + 1])]] += 1.0
        counts /= n_di
        return counts


# ---------------------------------------------------------------------------
# PseAAC
# ---------------------------------------------------------------------------

class PseAAC(Featurizer):
    """Pseudo amino acid composition (Chou 2001) — (20 + λ)-dimensional vector.

    Combines the standard amino acid composition with *λ* sequence-order
    correlation factors that encode long-range physicochemical information
    along the chain.

    The correlation factor of rank *k* is::

        θ_k = 1/(L-k) * Σ_{i=1}^{L-k} Θ(r_i, r_{i+k})

    where Θ is the average squared difference in normalised physicochemical
    properties (hydrophobicity, hydrophilicity, side-chain mass) between
    residues *i* and *i+k*.

    The final vector entries are::

        p_i        = f_i          / (1 + w * Σ_k θ_k),  i = 1 … 20
        p_{20+k}   = w * θ_k     / (1 + w * Σ_k θ_k),  k = 1 … λ

    where *f_i* is the raw amino-acid frequency.

    Parameters
    ----------
    lambda_ : int
        Number of pseudo components (sequence depth). Must satisfy
        ``lambda_ < sequence length``. Defaults to 30.
    weight : float
        Relative weight *w* for the pseudo components. Defaults to 0.05.

    References
    ----------
    Chou, K.-C. (2001). Prediction of protein cellular attributes using
    pseudo-amino acid composition. *Proteins*, 43(3), 246–255.

    Example
    -------
    >>> feat = PseAAC(lambda_=10)
    >>> vec = feat.extract_one("ACDEFGHIKLMNPQRSTVWY")
    >>> vec.shape
    (30,)
    >>> round(float(vec.sum()), 6)
    1.0
    """

    name = "pseaac"

    def __init__(self, lambda_: int = 30, weight: float = 0.05):
        if lambda_ < 1:
            raise ValueError("lambda_ must be >= 1.")
        if weight < 0:
            raise ValueError("weight must be non-negative.")
        self.lambda_ = lambda_
        self.weight = weight

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        seq = _filter_seq(seq)
        L = len(seq)
        lam = min(self.lambda_, L - 1)  # guard against short sequences

        dim = 20 + lam
        if L == 0 or lam == 0:
            vec = np.zeros(20 + self.lambda_, dtype=np.float32)
            if L > 0:
                # Only composition part, no pseudo components available
                for aa in seq:
                    vec[_AA_INDEX[aa]] += 1.0 / L
            return vec

        # --- amino acid frequencies ---
        f = np.zeros(20, dtype=np.float64)
        for aa in seq:
            f[_AA_INDEX[aa]] += 1.0
        f /= L

        # --- property values for each position ---
        # shape (L, 3): normalised properties at each residue
        H = np.array([_NORM_PROPS[:, _AA_INDEX[aa]] for aa in seq])  # (L, 3)

        # --- sequence-order correlation factors ---
        theta = np.zeros(lam, dtype=np.float64)
        for k in range(1, lam + 1):
            diff = H[:L - k] - H[k:]        # (L-k, 3)
            theta[k - 1] = np.mean(np.mean(diff ** 2, axis=1))

        # --- normalised PseAAC vector ---
        denom = 1.0 + self.weight * theta.sum()
        vec = np.empty(20 + lam, dtype=np.float32)
        vec[:20] = f / denom
        vec[20:20 + lam] = (self.weight * theta / denom).astype(np.float32)

        # Pad to requested size if lam < lambda_ (short sequence)
        if lam < self.lambda_:
            full = np.zeros(20 + self.lambda_, dtype=np.float32)
            full[:20 + lam] = vec
            return full

        return vec
