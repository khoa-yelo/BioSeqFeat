"""
BioSeqFeat: Feature extraction for biological sequences.

Public API
----------
Featurizer     -- abstract base class for all feature extractors
Pipeline       -- chains multiple featurizers with optional weighting

Subpackages
-----------
bioseqfeat.protein  -- protein sequence featurizers (BlosumAvg, BlosumCompress)
bioseqfeat.dna      -- DNA sequence featurizers (planned)
bioseqfeat.rna      -- RNA sequence featurizers (planned)
"""

from .base import Featurizer, Pipeline
from .protein import BlosumAvg, BlosumCompress

__all__ = ["Featurizer", "Pipeline", "BlosumAvg", "BlosumCompress"]
