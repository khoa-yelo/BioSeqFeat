"""
High-level Featurizer classes backed by BLOSUM62 feature extraction.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..base import Featurizer
from .blosum import average_embedding, compress_sequence


class BlosumAvg(Featurizer):
    """Mean BLOSUM62 embedding across all residues.

    Produces a single (20,) vector regardless of sequence length by averaging
    the BLOSUM62 row vectors for each residue.

    Example
    -------
    >>> feat = BlosumAvg()
    >>> vec = feat.extract_one("ACDEFGHIKLMNPQRSTVWY")
    >>> vec.shape
    (20,)
    """

    name = "blosum_avg"

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        return average_embedding(seq)


class BlosumCompress(Featurizer):
    """Position-aware BLOSUM62 compression to a fixed-size vector.

    Compresses a variable-length sequence to a ``(dim * 20,)`` vector by
    pooling BLOSUM62 embeddings into *dim* positional bins.

    Parameters
    ----------
    dim : int
        Number of positional bins. Output length is ``dim * 20``.
    method : {"moving_avg", "adaptive_pool", "dct"}
        Pooling strategy. Defaults to ``"dct"``.

    Example
    -------
    >>> feat = BlosumCompress(dim=10, method="dct")
    >>> vec = feat.extract_one("ACDEFGHIKLMNPQRSTVWY")
    >>> vec.shape
    (200,)
    """

    name = "blosum_compress"

    def __init__(
        self,
        dim: int = 20,
        method: Literal["moving_avg", "adaptive_pool", "dct"] = "dct",
    ):
        self.dim = dim
        self.method = method

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        return compress_sequence(seq, dim=self.dim, method=self.method)
