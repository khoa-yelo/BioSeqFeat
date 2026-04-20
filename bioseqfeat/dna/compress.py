"""
Position-compressed nucleotide encoding for DNA sequences.

Classes
-------
KmerCompress -- fixed-length positional encoding via DCT compression → (dim × 4,)

Background
----------
Analogous to BlosumCompress for proteins.  Each nucleotide is encoded as a
4-dimensional one-hot vector {A:[1,0,0,0], C:[0,1,0,0], G:[0,0,1,0], T:[0,0,0,1]}.
The resulting (L, 4) matrix is compressed along the sequence dimension into a
fixed (dim, 4) representation using a Discrete Cosine Transform (DCT), then
flattened to (dim * 4,).

The DCT preserves the most important frequency components of the positional
one-hot signal, capturing both local and global nucleotide patterns in a
position-aware, length-invariant way.
"""

from __future__ import annotations

import numpy as np

from ..base import Featurizer
from .composition import NUCLEOTIDES, _filter_seq

_NT_INDEX = {nt: i for i, nt in enumerate(NUCLEOTIDES)}  # A=0, C=1, G=2, T=3


def _onehot_encode(seq: str) -> np.ndarray:
    """One-hot encode a canonical DNA sequence → (L, 4) float32 array."""
    L = len(seq)
    mat = np.zeros((L, 4), dtype=np.float32)
    for i, nt in enumerate(seq):
        mat[i, _NT_INDEX[nt]] = 1.0
    return mat


def _dct_compress(mat: np.ndarray, dim: int) -> np.ndarray:
    """Compress a (L, 4) matrix to (dim, 4) using a real DCT along axis 0.

    Uses the Type-II DCT (scipy convention) and keeps the first *dim*
    coefficients, normalised to have comparable magnitudes regardless of L.
    Falls back to zero-padding if L < dim.
    """
    from scipy.fft import dct as scipy_dct

    L = mat.shape[0]
    if L == 0:
        return np.zeros((dim, 4), dtype=np.float32)

    if L < dim:
        # Zero-pad to dim rows before DCT
        padded = np.zeros((dim, 4), dtype=np.float64)
        padded[:L] = mat
        mat_for_dct = padded
    else:
        mat_for_dct = mat.astype(np.float64)

    # Apply DCT column-wise
    coeffs = scipy_dct(mat_for_dct, type=2, axis=0, norm="ortho")  # (L_or_dim, 4)
    return coeffs[:dim].astype(np.float32)


class KmerCompress(Featurizer):
    """Position-compressed nucleotide encoding — (dim × 4)-dimensional vector.

    Encodes each nucleotide as a 4-channel one-hot vector, then compresses
    the (L, 4) positional matrix to a fixed (dim, 4) representation via
    a Discrete Cosine Transform, and flattens to (dim*4,).

    This is order-aware (position matters) and length-invariant (output is
    always dim*4 regardless of sequence length).

    Parameters
    ----------
    dim : int
        Number of DCT coefficients to retain per channel.  Defaults to 20,
        giving a 80-dimensional output.

    Example
    -------
    >>> feat = KmerCompress(dim=20)
    >>> vec = feat.extract_one("ATGCATGCATGC")
    >>> vec.shape
    (80,)
    """

    name = "kmer_compress"

    def __init__(self, dim: int = 20):
        if dim < 1:
            raise ValueError("dim must be >= 1.")
        self.dim = dim

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        seq = _filter_seq(seq)
        if not seq:
            return np.zeros(self.dim * 4, dtype=np.float32)
        mat = _onehot_encode(seq)
        compressed = _dct_compress(mat, self.dim)     # (dim, 4)
        return compressed.flatten()                    # (dim*4,)
