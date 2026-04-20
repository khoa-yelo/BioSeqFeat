"""
DNA sequence feature extractors.

Featurizers
-----------
MNC              -- mono-nucleotide composition                  →  (4,)
DNC              -- di-nucleotide composition                    → (16,)
TNC              -- tri-nucleotide composition                   → (64,)
GlobalDescriptors -- GC/AT skews, CpG O/E, Tm, entropy, etc.   → (12,)
CTD              -- Composition/Transition/Distribution          → (39,)
PseKNC           -- pseudo K-mer nucleotide composition          → (16 + λ,)
KmerCompress     -- DCT-compressed positional one-hot encoding  → (dim × 4,)
"""

from .composition import MNC, DNC, TNC
from .global_descriptors import GlobalDescriptors
from .ctd import CTD
from .pseudo_knc import PseKNC
from .compress import KmerCompress

__all__ = [
    "MNC",
    "DNC",
    "TNC",
    "GlobalDescriptors",
    "CTD",
    "PseKNC",
    "KmerCompress",
]
