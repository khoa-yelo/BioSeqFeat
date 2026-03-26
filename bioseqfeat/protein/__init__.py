"""
Protein sequence feature extractors.

Classes
-------
BlosumAvg      -- mean BLOSUM62 embedding per residue → (20,) vector
BlosumCompress -- position-aware BLOSUM62 compression → (dim*20,) vector
"""

from .featurizers import BlosumAvg, BlosumCompress

__all__ = ["BlosumAvg", "BlosumCompress"]
