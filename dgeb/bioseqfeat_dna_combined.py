"""
Combined BioSeqFeat pipeline for DNA DGEB evaluation.

All seven featurizers are L2-normalized to unit norm before concatenation,
so Pipeline weights directly control each block's contribution to cosine
similarity (same sqrt(w) scaling trick as the protein combined pipeline).

Feature set and weight rationale
---------------------------------
Featurizer              Dim     Weight  Fraction  Rationale
KmerCompress(dim=20)     80     5.0      30 %     Position-aware; captures local motifs & order
TNC                      64     4.0      24 %     Codon-level context; richest composition signal
DNC                      16     3.0      18 %     Dinucleotide stacking; well-validated in literature
PseKNC(λ=20)             36     3.0      18 %     Long-range order from dinucleotide physicochemistry
CTD                      39     2.0      12 %     Sequential topology (transitions, distributions)
MNC                       4     1.0       6 %     Baseline GC/AT balance; partially redundant
GlobalDescriptors        12     0.5       3 %     Bulk properties (skews, CpG, entropy, Tm)
                               ----
Total embedding dim: 251        18.5    ~111 % (before re-normalisation)

Design notes
------------
* KmerCompress and TNC carry the heaviest weights because position-order and
  trinucleotide context are the strongest handcrafted DNA signals.
* PseKNC augments DNC with physicochemical sequence-order correlations (analogous
  to PseAAC for protein); weighted equally with DNC.
* CTD encodes sequential topology (purine/pyrimidine runs, transitions) at lower
  weight than the raw k-mer features.
* MNC is kept at 1.0 as a baseline; it is largely captured by DNC/TNC.
* GlobalDescriptors gets 0.5 — orthogonal bulk properties (CpG O/E, skews,
  Shannon entropy) are useful but low-dimensional and partially redundant.
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

from bioseqfeat import Pipeline, NormalizedFeaturizer
from bioseqfeat.dna import MNC, DNC, TNC, GlobalDescriptors, CTD, PseKNC, KmerCompress

_LAMBDA = 20

_FEATURIZERS = [
    NormalizedFeaturizer(KmerCompress(dim=20)),
    NormalizedFeaturizer(TNC()),
    NormalizedFeaturizer(DNC()),
    NormalizedFeaturizer(PseKNC(lambda_=_LAMBDA)),
    NormalizedFeaturizer(CTD()),
    NormalizedFeaturizer(MNC()),
    NormalizedFeaturizer(GlobalDescriptors()),
]

_WEIGHTS = [5.0, 4.0, 3.0, 3.0, 2.0, 1.0, 0.5]

# dim = 80 + 64 + 16 + (16 + 20) + 39 + 4 + 12 = 251
_FEAT_DIM = 80 + 64 + 16 + (16 + _LAMBDA) + 39 + 4 + 12

_PIPELINE = Pipeline(featurizers=_FEATURIZERS, weights=_WEIGHTS)


class _BioSeqFeatEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, *args, **kwargs):
        return None


class _BioSeqFeatTokenizer:
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


class BioSeqFeatDNACombinedTransformer(BioSeqTransformer):
    """DGEB-compatible model combining all seven BioSeqFeat DNA featurizers.

    Each featurizer is L2-normalized to unit norm before concatenation.
    Total embedding dimension: 251.

    Example
    -------
    >>> model = BioSeqFeatDNACombinedTransformer()
    >>> evaluation = dgeb.DGEB(tasks=dgeb.get_tasks_by_modality(dgeb.Modality.DNA))
    >>> results = evaluation.run(model, output_folder="results/dna_combined")
    """

    MODEL_NAMES = ["bioseqfeat-dna-combined"]

    def __init__(self, **kwargs):
        self._feat_dim = _FEAT_DIM
        self._pipeline = _PIPELINE
        self._tokenizer_instance = _BioSeqFeatTokenizer()
        kwargs.setdefault("num_processes", 0)
        super().__init__(model_name="bioseqfeat-dna-combined", **kwargs)

    def _load_model(self, model_name):
        return _BioSeqFeatEncoder(self._feat_dim)

    def _get_tokenizer(self, model_name):
        return self._tokenizer_instance

    def _encode_single_batch(self, batch_dict):
        sequences = self._tokenizer_instance._last_sequences
        features = self._pipeline.extract_batch(sequences)  # (batch_size, feat_dim)
        features_tensor = torch.from_numpy(features).float().to(self.device)
        return features_tensor.unsqueeze(1).expand(-1, len(self.layers), -1).contiguous()

    @property
    def num_layers(self) -> int:
        return 1

    @property
    def embed_dim(self) -> int:
        return self._feat_dim

    @property
    def modality(self):
        return Modality.DNA


if __name__ == "__main__":
    model = BioSeqFeatDNACombinedTransformer()
    evaluation = dgeb.DGEB(tasks=dgeb.get_tasks_by_modality(dgeb.Modality.DNA))
    results = evaluation.run(model, output_folder="results/dna_combined")
    print(results)
