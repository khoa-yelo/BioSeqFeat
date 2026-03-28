import torch
import numpy as np
from types import SimpleNamespace
import dgeb
from dgeb.models import BioSeqTransformer
from dgeb.tasks.tasks import Modality


class _DummyEncoder(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=embed_dim)

    def forward(self, *args, **kwargs):
        return None


class _DummyTokenizer:
    def __call__(self, sequences, max_length=None, padding=True, truncation=True):
        batch_size = len(sequences)
        seq_len = max_length if max_length is not None else 10
        return {
            "input_ids": torch.zeros((batch_size, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        }


class RandomBioSeqTransformer(BioSeqTransformer):
    MODEL_NAMES = ["random-model"]

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 2,
        **kwargs,
    ):
        self._embed_dim = embed_dim
        self._num_layers = num_layers
        super().__init__(model_name="random-model", **kwargs)

    def _load_model(self, model_name):
        return _DummyEncoder(self._embed_dim)

    def _get_tokenizer(self, model_name):
        return _DummyTokenizer()

    def _encode_single_batch(self, batch_dict):
        batch_size = batch_dict["input_ids"].shape[0]
        embeds = torch.randn(
            batch_size,
            len(self.layers),
            self.embed_dim,
            device=self.device,
        )
        return embeds

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def modality(self):
        return Modality.PROTEIN

if __name__ == "__main__":
    model = RandomBioSeqTransformer()
    evaluation = dgeb.DGEB(tasks=dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN))
    results = evaluation.run(model, output_folder="results/random_baseline")
    print(results)