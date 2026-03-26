from __future__ import annotations
 
from typing import Literal
 
import numpy as np
 
from protein_feature_extractors import average_embedding, compress_sequence
from core import Featurizer
 
 
class BlosumAvg(Featurizer):
    """Mean BLOSUM embedding per residue → (20,) vector."""
 
    name = "blosum_avg"
 
    def __init__(self, matrix_name: str = "BLOSUM62"):
        self.matrix_name = matrix_name
 
    def extract_one(self, seq: str, **kwargs) -> np.ndarray:
        emb = average_embedding(seq)  # (L, 20)
        return emb
 
class BlosumCompress(Featurizer):
    """Position-aware BLOSUM compression → (dim*20,) vector."""
 
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