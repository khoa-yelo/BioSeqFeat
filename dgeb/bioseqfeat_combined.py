"""
Combined BioSeqFeat pipeline for DGEB evaluation.

All eight featurizers are L2-normalized to unit norm before concatenation,
making the Pipeline weights directly control each block's contribution to
cosine similarity:

    cosine(combined) ≈ Σ_i w_i · cosine_i(v1, v2) / Σ_i w_i

Without per-block normalization the raw scale differences make most
featurizers invisible: BlosumCompress has norm ~35 while probability-based
featurizers (AAC, DPC, PseAAC) have norm ~0.1, a 300× gap. This is why
the previous equal-weight BLOSUM+composition run matched BLOSUM alone.

Weight rationale
----------------
Featurizer              Dim    Weight  Fraction  Rationale
BlosumAvg               20     5.0     25 %      Proven best performer; evolutionary composition
BlosumCompress(dct)     400    5.0     25 %      Adds sequence-order to evolutionary signal
HMMProfile(Pfam)        1011   3.0     15 %      Domain/family membership; highly discriminative per-seq signal
DPC                     400    2.0     10 %      Actual AA-pair frequencies; captures order
CTD                     147    2.0     10 %      Unique topology: transitions + distributions
PseAAC(λ=30)            50     1.5      8 %      Long-range physicochemical correlations
AAC                     20     1.0      5 %      Baseline composition; partially redundant
GlobalDescriptors       9      0.5      3 %      Orthogonal bulk properties (MW, pI, 2° struct)
                               ----
Total embedding dim: 2057      20.0    100 %
  (1046 static + 1011 HMM profiles; HMM dim is determined at runtime from the DB)

BLOSUM features receive 50 % of influence. HMM domain membership adds
complementary evolutionary signal (family-level, not residue-level) at 15 %.

Requires HMMER3 (hmmscan) on PATH and the pressed HMM database:
    hmmpress db/Pfam-A.subset.hmm
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from types import SimpleNamespace
import dgeb
from dgeb.models import BioSeqTransformer
from dgeb.tasks.tasks import Modality

from bioseqfeat import Pipeline, NormalizedFeaturizer, BlosumAvg, BlosumCompress
from bioseqfeat.protein.composition import AAC, DPC, PseAAC
from bioseqfeat.protein.ctd import CTD
from bioseqfeat.protein.global_descriptors import GlobalDescriptors
from bioseqfeat.protein.hmm import HMMFeaturizer

_LAMBDA = 30
# Static dims: 20 + 400 + 400 + 147 + 50 + 20 + 9 = 1046
# HMM dim (1011) is determined at runtime from the DB file.
_STATIC_DIM = 20 + 20 * 20 + 400 + 147 + (20 + _LAMBDA) + 20 + 9

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "db", "Pfam-A.subset.hmm",
)
_E_THRESH = 10.0  # loose threshold — keeps weak hits, avoids all-zero rows
_CPU = 4

# Static featurizers (cheap to construct; no file I/O)
_STATIC_FEATURIZERS = [
    NormalizedFeaturizer(BlosumAvg()),
    NormalizedFeaturizer(BlosumCompress(dim=20, method="dct")),
    NormalizedFeaturizer(DPC()),
    NormalizedFeaturizer(CTD()),
    NormalizedFeaturizer(PseAAC(lambda_=_LAMBDA)),
    NormalizedFeaturizer(AAC()),
    NormalizedFeaturizer(GlobalDescriptors()),
]
_STATIC_WEIGHTS = [5.0, 5.0, 2.0, 2.0, 1.5, 1.0, 0.5]
_HMM_WEIGHT = 3.0


class _BioSeqFeatEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, *args, **kwargs):
        return None


class _BioSeqFeatTokenizer:
    """Dummy tokenizer that stores raw sequences for use in encoding."""

    def __init__(self):
        self._last_sequences: list[str] = []

    def __call__(self, sequences, max_length=None, padding=True, truncation=True):
        self._last_sequences = list(sequences)
        batch_size = len(sequences)
        seq_len = max_length if max_length is not None else 10
        return {
            "input_ids": torch.zeros((batch_size, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        }


class BioSeqFeatCombinedTransformer(BioSeqTransformer):
    """DGEB-compatible model combining all eight BioSeqFeat featurizers.

    Each featurizer is L2-normalized to unit norm before concatenation.
    Weights are chosen to give BLOSUM features 50 % of influence, HMM
    domain profiles 15 %, and the remaining featurizers meaningful signal.

    The HMM profile DB is loaded on construction (reads the .hmm file once
    to determine profile count); total embedding dimension is therefore
    determined at runtime (1046 static + n_profiles HMM = 2057 for the
    default 1011-profile Pfam clan-representative subset).

    Parameters
    ----------
    hmm_db : str
        Path to the pressed HMM database. Defaults to
        ``db/Pfam-A.subset.hmm`` relative to the repository root.
    e_thresh : float
        hmmscan E-value threshold (default 10.0).
    cpu : int
        Number of CPU threads for hmmscan (default 4).

    Example
    -------
    >>> model = BioSeqFeatCombinedTransformer()
    >>> evaluation = dgeb.DGEB(tasks=dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN))
    >>> results = evaluation.run(model, output_folder="results/bioseqfeat_combined_hmm")
    """

    MODEL_NAMES = ["bioseqfeat-combined-hmm"]

    def __init__(self, hmm_db: str = _DB_PATH, e_thresh: float = _E_THRESH,
                 cpu: int = _CPU, **kwargs):
        # HMM featurizer reads the DB file here (once) to learn n_profiles
        _hmm_raw = HMMFeaturizer(hmm_db, e_thresh=e_thresh, cpu=cpu)
        _hmm_norm = NormalizedFeaturizer(_hmm_raw)
        self._feat_dim = _STATIC_DIM + _hmm_raw.n_profiles
        self._pipeline = Pipeline(
            featurizers=_STATIC_FEATURIZERS + [_hmm_norm],
            weights=_STATIC_WEIGHTS + [_HMM_WEIGHT],
        )
        self._tokenizer_instance = _BioSeqFeatTokenizer()
        kwargs.setdefault("num_processes", 0)
        super().__init__(model_name="bioseqfeat-combined-hmm", **kwargs)

    def _load_model(self, model_name):
        return _BioSeqFeatEncoder(self._feat_dim)

    def _get_tokenizer(self, model_name):
        return self._tokenizer_instance

    def _encode_single_batch(self, batch_dict):
        sequences = self._tokenizer_instance._last_sequences
        features = self._pipeline.extract_batch(sequences)  # (batch_size, feat_dim)
        features_tensor = torch.from_numpy(features).float().to(self.device)
        # DGEB expects shape (batch_size, num_layers, embed_dim)
        return features_tensor.unsqueeze(1).expand(-1, len(self.layers), -1).contiguous()

    @property
    def num_layers(self) -> int:
        return 1

    @property
    def embed_dim(self) -> int:
        return self._feat_dim

    @property
    def modality(self):
        return Modality.PROTEIN


if __name__ == "__main__":
    model = BioSeqFeatCombinedTransformer()
    evaluation = dgeb.DGEB(tasks=dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN))
    results = evaluation.run(model, output_folder="results/bioseqfeat_combined_hmm")
    print(results)
