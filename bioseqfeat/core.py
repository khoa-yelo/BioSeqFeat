"""
Featurizer base class and Pipeline for biological sequence feature extraction.
 
Featurizer: single abstract method `extract_one` to implement.
Pipeline:  chains multiple featurizers, concatenates their output vectors.
"""
 
from __future__ import annotations
 
import abc
from typing import Sequence
 
import numpy as np
 
 
class Featurizer(abc.ABC):
    """
    Base class for all feature featurizers.
 
    Subclass contract: set `name` and implement `extract_one`.
    That's it. Everything else is optional.
    """
 
    name: str = "base"
 
    @abc.abstractmethod
    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        """seq → 1-D feature vector."""
        ...
 
    def extract_batch(self, seqs: Sequence[str], **kwargs) -> np.ndarray:
        """List of seqs → (N, D) array."""
        return np.stack([self.extract_one(s, **kwargs) for s in seqs])
 
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"
 
 
class Pipeline:
    """
    Chain multiple featurizers. Output is the weighted concatenation of
    all their vectors.
 
    Each featurizer's output is scaled by sqrt(weight) before concatenation.
    This means standard cosine similarity on the concatenated vector is
    equivalent to weighted cosine similarity over the individual blocks —
    no custom distance function needed.
 
    Usage
    -----
    pipe = Pipeline(
        featurizers=[BlosumAvg(), AAC(), SeqLen()],
        weights=[5.0, 1.0, 0.1],   # BLOSUM matters 50x more than length
    )
    vec = pipe.extract_one(seq)             # 1-D weighted-concat vector
    mat = pipe.extract_batch([s1, s2, s3])  # (N, total_dim)
 
    # standard cosine on these vectors == weighted cosine over feature blocks
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    """
 
    def __init__(
        self,
        featurizers: Sequence[Featurizer],
        weights: Sequence[float] | None = None,
    ):
        if not featurizers:
            raise ValueError("Pipeline needs at least one featurizer.")
        self.featurizers = list(featurizers)
 
        if weights is None:
            self.weights = np.ones(len(self.featurizers), dtype=np.float32)
        else:
            if len(weights) != len(self.featurizers):
                raise ValueError(
                    f"Got {len(weights)} weights for {len(self.featurizers)} featurizers."
                )
            self.weights = np.array(weights, dtype=np.float32)
 
        if (self.weights < 0).any():
            raise ValueError("Weights must be non-negative.")
 
        # sqrt for the scaling trick: cos(sqrt(w)*a, sqrt(w)*b) = weighted cosine
        self._scales = np.sqrt(self.weights)
 
    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        parts = []
        for ext, scale in zip(self.featurizers, self._scales):
            v = ext.extract_one(seq, **kwargs)
            parts.append(v * scale)
        return np.concatenate(parts)
 
    def extract_batch(self, seqs: Sequence[str], **kwargs) -> np.ndarray:
        return np.stack([self.extract_one(s, **kwargs) for s in seqs])
 
    @property
    def names(self) -> list[str]:
        return [ext.name for ext in self.featurizers]
 
    def __repr__(self) -> str:
        pairs = [f"{n}={w:.2g}" for n, w in zip(self.names, self.weights)]
        return f"Pipeline([{', '.join(pairs)}])"
