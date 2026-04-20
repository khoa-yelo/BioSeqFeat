"""
BioSeqFeat: Feature extraction for biological sequences.

Public API
----------
Featurizer     -- abstract base class for all feature extractors
Pipeline       -- chains multiple featurizers with optional weighting

Subpackages
-----------
bioseqfeat.protein  -- protein sequence featurizers (BlosumAvg, BlosumCompress, …)
bioseqfeat.dna      -- DNA sequence featurizers (MNC, DNC, TNC, CTD, PseKNC, …)
bioseqfeat.rna      -- RNA sequence featurizers (planned)
"""

from .base import Featurizer, NormalizedFeaturizer, Pipeline
from .protein import BlosumAvg, BlosumCompress
from .dna import MNC, DNC, TNC, KmerCompress

__all__ = [
    "Featurizer",
    "NormalizedFeaturizer",
    "Pipeline",
    # protein
    "BlosumAvg",
    "BlosumCompress",
    # dna
    "MNC",
    "DNC",
    "TNC",
    "KmerCompress",
]
