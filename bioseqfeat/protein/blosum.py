"""
Low-level BLOSUM62-based feature extraction functions for protein sequences.

Module-level constants
----------------------
AA_ORDER  -- list of 20 standard amino acids in canonical order
AA_TO_IDX -- dict mapping each amino acid letter to its row/column index
BLOSUM62  -- (20, 20) float32 numpy array of BLOSUM62 substitution scores
"""

from typing import Literal

import numpy as np
from Bio.Align import substitution_matrices

# ---------------------------------------------------------------------------
# Build BLOSUM62 lookup table at import time
# ---------------------------------------------------------------------------

_BLOSUM62_BIO = substitution_matrices.load("BLOSUM62")

# Keep only the 20 standard amino acids (skip ambiguous codes B, Z, X, *)
AA_ORDER: list[str] = [aa for aa in _BLOSUM62_BIO.alphabet if aa in "ARNDCQEGHILKMFPSTWYV"]
AA_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(AA_ORDER)}
EMBED_DIM: int = len(AA_ORDER)  # 20

BLOSUM62: np.ndarray = np.array(
    [[_BLOSUM62_BIO[a, b] for b in AA_ORDER] for a in AA_ORDER],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _seq_to_embeddings(seq: str) -> np.ndarray:
    """Convert a protein sequence to an (L, 20) array of BLOSUM62 row vectors.

    Non-standard residues (X, B, Z, U, etc.) are silently skipped.

    Raises
    ------
    ValueError
        If no valid standard amino acids are found in the sequence.
    """
    rows = [BLOSUM62[idx] for aa in seq.upper() if (idx := AA_TO_IDX.get(aa)) is not None]
    if not rows:
        raise ValueError(f"No valid amino acids found in sequence: {seq[:30]!r}")
    return np.stack(rows)  # (L, 20)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def average_embedding(seq: str) -> np.ndarray:
    """Return the mean BLOSUM62 embedding across all residues.

    Parameters
    ----------
    seq : str
        Protein sequence (case-insensitive).

    Returns
    -------
    np.ndarray, shape (20,)
    """
    emb = _seq_to_embeddings(seq)   # (L, 20)
    return emb.mean(axis=0)         # (20,)


def compress_sequence(
    seq: str,
    dim: int = 20,
    method: Literal["moving_avg", "adaptive_pool", "dct"] = "adaptive_pool",
) -> np.ndarray:
    """Compress a variable-length sequence to a fixed-size feature vector.

    Sequences shorter than *dim* are zero-padded. Longer sequences are
    downsampled using the chosen *method*.

    Parameters
    ----------
    seq : str
        Protein sequence (case-insensitive).
    dim : int
        Number of output bins. Final vector length is ``dim * 20``.
    method : {"moving_avg", "adaptive_pool", "dct"}
        Compression strategy (see Notes).

    Returns
    -------
    np.ndarray, shape (dim * 20,)

    Notes
    -----
    * ``moving_avg``    -- divide into *dim* equal bins, average each bin.
    * ``adaptive_pool`` -- weighted average with fractional boundary handling,
                          analogous to PyTorch's ``AdaptiveAvgPool1d``.
    * ``dct``           -- discrete cosine transform; keep top *dim* coefficients.
    """
    emb = _seq_to_embeddings(seq)   # (L, 20)
    L = emb.shape[0]

    if L <= dim:
        padded = np.zeros((dim, EMBED_DIM), dtype=np.float32)
        padded[:L] = emb
        return padded.ravel()

    if method == "moving_avg":
        return _moving_avg_pool(emb, dim).ravel()
    elif method == "adaptive_pool":
        return _adaptive_avg_pool(emb, dim).ravel()
    elif method == "dct":
        return _dct_compress(emb, dim).ravel()
    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose from 'moving_avg', 'adaptive_pool', 'dct'."
        )


# ---------------------------------------------------------------------------
# Compression backends (private)
# ---------------------------------------------------------------------------

def _moving_avg_pool(emb: np.ndarray, dim: int) -> np.ndarray:
    """Divide sequence into *dim* equal bins; return the mean of each bin."""
    L = emb.shape[0]
    out = np.zeros((dim, EMBED_DIM), dtype=np.float32)
    for i in range(dim):
        bin_start = int(i * L / dim)
        bin_end = int((i + 1) * L / dim)
        out[i] = emb[bin_start:bin_end].mean(axis=0)
    return out


def _adaptive_avg_pool(emb: np.ndarray, dim: int) -> np.ndarray:
    """Weighted average pooling with fractional boundary handling.

    Boundary residues that fall partially inside a bin are weighted by their
    fractional overlap, matching PyTorch's ``AdaptiveAvgPool1d`` semantics.
    """
    L = emb.shape[0]
    out = np.zeros((dim, EMBED_DIM), dtype=np.float32)

    for i in range(dim):
        bin_start = i * L / dim
        bin_end = (i + 1) * L / dim

        win_start = int(np.floor(bin_start))
        win_end = int(np.ceil(bin_end))
        if win_start == win_end:
            win_end = win_start + 1

        window = emb[win_start:win_end].copy()

        # Down-weight the left boundary residue by its fractional overlap
        left_fraction = bin_start - win_start
        if left_fraction > 0:
            window[0] *= 1.0 - left_fraction

        # Down-weight the right boundary residue by its fractional overlap
        right_fraction = win_end - bin_end
        if right_fraction > 0 and len(window) > 0:
            window[-1] *= 1.0 - right_fraction

        bin_weight = bin_end - bin_start
        out[i] = window.sum(axis=0) / bin_weight

    return out


def _dct_compress(emb: np.ndarray, dim: int) -> np.ndarray:
    """Keep the top *dim* DCT-II coefficients along the sequence axis."""
    try:
        from scipy.fft import dct
    except ImportError:
        from scipy.fftpack import dct

    coeffs = dct(emb, type=2, axis=0, norm="ortho")    # (L, 20)
    return coeffs[:dim].astype(np.float32)              # (dim, 20)
