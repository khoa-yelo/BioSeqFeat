"""
NucleotideTransformer v2 (50M, multi-species) benchmark for DNA DGEB evaluation.

Model: InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
  - 50M parameter transformer pretrained on 850 multi-species genomes
  - Input: nucleotide sequences tokenised as 6-mers (k=6)
  - Output: per-token hidden states; mean-pooled for sequence embedding

This serves as the neural baseline to compare against BioSeqFeat handcrafted features.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cache HuggingFace models on scratch to avoid filling home quota
os.environ.setdefault("HF_HOME", "/scratch/users/khoang99/hf_cache")

import torch
import numpy as np
from types import SimpleNamespace
import dgeb
from dgeb.models import BioSeqTransformer
from dgeb.tasks.tasks import Modality
from transformers import AutoTokenizer, AutoModel

_MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
_MAX_LENGTH = 2048  # max tokens (= 6 * 2048 = 12,288 nucleotides)


class NucleotideTransformerV2(BioSeqTransformer):
    """DGEB-compatible wrapper for NucleotideTransformer-v2-50m-multi-species.

    Encodes DNA sequences using mean-pooling over all non-padding token
    hidden states from the last (or specified) transformer layer(s).

    Parameters
    ----------
    model_name : str
        HuggingFace model ID. Defaults to the 50M multi-species checkpoint.
    layers : list[int] | None
        Which hidden layers to expose.  Defaults to [12] (last layer of
        the 50M model which has 12 transformer blocks).

    Example
    -------
    >>> model = NucleotideTransformerV2()
    >>> evaluation = dgeb.DGEB(tasks=dgeb.get_tasks_by_modality(dgeb.Modality.DNA))
    >>> results = evaluation.run(model, output_folder="results/dna_nt_v2_50m")
    """

    MODEL_NAMES = [_MODEL_NAME]

    def __init__(self, model_name: str = _MODEL_NAME, **kwargs):
        kwargs.setdefault("num_processes", 0)
        super().__init__(model_name=model_name, **kwargs)

    def _load_model(self, model_name: str):
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        return model

    def _get_tokenizer(self, model_name: str):
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def _encode_single_batch(self, batch_dict):
        input_ids = batch_dict["input_ids"].to(self.device)
        attention_mask = batch_dict["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # hidden_states: tuple of (batch, seq_len, hidden) for each layer
        # layers is 1-indexed in DGEB convention; hidden_states[0] = embedding layer
        hidden_states = outputs.hidden_states  # (n_layers+1, batch, seq, hidden)

        # Collect requested layers and mean-pool over non-padding tokens
        layer_embeddings = []
        for layer_idx in self.layers:
            hs = hidden_states[layer_idx]  # (batch, seq_len, hidden)
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
            layer_embeddings.append(pooled)  # (batch, hidden)

        # Stack to (batch, num_layers, hidden)
        result = torch.stack(layer_embeddings, dim=1)
        return result

    @property
    def num_layers(self) -> int:
        # NucleotideTransformer-v2-50m has 12 transformer layers
        return 12

    @property
    def embed_dim(self) -> int:
        # Hidden size of 50M model
        return 512

    @property
    def modality(self):
        return Modality.DNA


if __name__ == "__main__":
    model = NucleotideTransformerV2()
    evaluation = dgeb.DGEB(tasks=dgeb.get_tasks_by_modality(dgeb.Modality.DNA))
    results = evaluation.run(model, output_folder="results/dna_nt_v2_50m")
    print(results)
