"""
Abstract base class and Pipeline for biological sequence feature extraction.

Classes
-------
Featurizer           -- abstract base; implement ``extract_one`` to create a new extractor.
NormalizedFeaturizer -- wraps any Featurizer and L2-normalizes its output to unit norm.
Pipeline             -- chains multiple featurizers with optional per-featurizer weighting.
"""

from __future__ import annotations

import abc
from typing import Sequence

import numpy as np


class Featurizer(abc.ABC):
    """Abstract base class for all sequence feature extractors.

    Subclass contract
    -----------------
    * Set a ``name`` class attribute (used for display and debugging).
    * Implement ``extract_one(seq) -> np.ndarray`` returning a 1-D vector.

    ``extract_batch`` is provided for free and handles any sequence iterable.
    """

    name: str = "base"

    @abc.abstractmethod
    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        """Extract a 1-D feature vector from a single sequence."""
        ...

    def extract_batch(self, seqs: Sequence[str], **kwargs) -> np.ndarray:
        """Extract features from multiple sequences.

        Parameters
        ----------
        seqs : sequence of str
            Input sequences.

        Returns
        -------
        np.ndarray, shape (N, D)
        """
        return np.stack([self.extract_one(s, **kwargs) for s in seqs])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"


class NormalizedFeaturizer(Featurizer):
    """Wraps a Featurizer and L2-normalizes its output to unit norm.

    This is essential when combining featurizers whose outputs have very
    different natural scales (e.g., BLOSUM embeddings with norm ~30 vs.
    probability distributions with norm ~0.1). Without normalization the
    Pipeline's ``sqrt(weight)`` scaling is meaningless because the raw
    norms already differ by orders of magnitude.

    After wrapping, each block has unit norm, so the Pipeline weights
    directly control each block's contribution to cosine similarity:

        cosine(combined) ≈ Σ_i w_i · cosine_i(v1, v2) / Σ_i w_i

    Parameters
    ----------
    featurizer : Featurizer
        The featurizer to wrap.
    eps : float
        Small constant added to the norm to avoid division by zero for
        zero vectors (e.g. empty or all-unknown sequences).
    """

    def __init__(self, featurizer: Featurizer, eps: float = 1e-8):
        self._featurizer = featurizer
        self._eps = eps

    @property
    def name(self) -> str:
        return f"norm_{self._featurizer.name}"

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        v = self._featurizer.extract_one(seq, **kwargs)
        norm = np.linalg.norm(v)
        return v / (norm + self._eps)

    def __repr__(self) -> str:
        return f"NormalizedFeaturizer({self._featurizer!r})"


class Pipeline:
    """Chain multiple featurizers into a single weighted feature vector.

    Each featurizer's output is scaled by ``sqrt(weight)`` before concatenation.
    This means that standard cosine similarity on the concatenated vector is
    equivalent to weighted cosine similarity over the individual feature blocks —
    no custom distance function needed.

    Parameters
    ----------
    featurizers : sequence of Featurizer
        Extractors to chain. At least one is required.
    weights : sequence of float, optional
        Per-featurizer weights (must be non-negative). Defaults to all ones.

    Example
    -------
    >>> pipe = Pipeline(
    ...     featurizers=[BlosumAvg(), BlosumCompress()],
    ...     weights=[5.0, 1.0],
    ... )
    >>> vec = pipe.extract_one("ACDEFGHIKLM")
    >>> mat = pipe.extract_batch(["ACDE", "FGHIK", "LMNPQ"])  # shape (3, D)
    """

    def __init__(
        self,
        featurizers: Sequence[Featurizer],
        weights: Sequence[float] | None = None,
    ):
        if not featurizers:
            raise ValueError("Pipeline requires at least one featurizer.")
        self.featurizers = list(featurizers)

        if weights is None:
            self.weights = np.ones(len(self.featurizers), dtype=np.float32)
        else:
            if len(weights) != len(self.featurizers):
                raise ValueError(
                    f"Got {len(weights)} weights but {len(self.featurizers)} featurizers."
                )
            self.weights = np.array(weights, dtype=np.float32)

        if (self.weights < 0).any():
            raise ValueError("All weights must be non-negative.")

        # Precompute scales: scaling by sqrt(w) makes cosine(v1, v2) == weighted cosine
        self._scales = np.sqrt(self.weights)

    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        """Extract and concatenate weighted features from a single sequence."""
        parts = [
            ext.extract_one(seq, **kwargs) * scale
            for ext, scale in zip(self.featurizers, self._scales)
        ]
        return np.concatenate(parts)

    def extract_batch(self, seqs: Sequence[str], **kwargs) -> np.ndarray:
        """Extract features from multiple sequences.

        Returns
        -------
        np.ndarray, shape (N, total_dim)
        """
        return np.stack([self.extract_one(s, **kwargs) for s in seqs])

    @property
    def names(self) -> list[str]:
        """Names of all featurizers in the pipeline."""
        return [ext.name for ext in self.featurizers]

    def __repr__(self) -> str:
        pairs = [f"{n}={w:.2g}" for n, w in zip(self.names, self.weights)]
        return f"Pipeline([{', '.join(pairs)}])"
