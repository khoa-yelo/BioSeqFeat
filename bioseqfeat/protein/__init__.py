"""
Protein sequence feature extractors.

Classes
-------
BlosumAvg         -- mean BLOSUM62 embedding per residue → (20,) vector
BlosumCompress    -- position-aware BLOSUM62 compression → (dim*20,) vector
AAC               -- amino acid composition → (20,) frequency vector
DPC               -- dipeptide composition  → (400,) frequency vector
PseAAC            -- pseudo amino acid composition (Chou 2001) → (20+λ,) vector
GlobalDescriptors -- 9 global physicochemical descriptors → (9,) vector
CTD               -- composition/transition/distribution → (147,) vector
HMMFeaturizer     -- hmmscan bit-score vector against HMM subset → (n_profiles,) vector
MMseqsLandmark    -- MMseqs2 bit-score vector against anchor proteins → (n_anchors,) vector
"""

from .featurizers import BlosumAvg, BlosumCompress
from .composition import AAC, DPC, PseAAC
from .global_descriptors import GlobalDescriptors
from .ctd import CTD
from .hmm import HMMFeaturizer
from .mmseqs import MMseqsLandmark

__all__ = ["BlosumAvg", "BlosumCompress", "AAC", "DPC", "PseAAC", "GlobalDescriptors", "CTD", "HMMFeaturizer", "MMseqsLandmark"]
