"""RoBERTa / DeBERTa sarcasm encoder.

Sequence embeddings are projected to 768 when the backbone width differs.
A checkpoint is marked ``trained: true`` only by the training script after a
real optimisation run. Loading the pretrained backbone alone does not set
that flag, and the inference API will not present those logits as a result.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from shared.constants import TEXT_HIDDEN
from shared.errors import CheckpointNotFoundError, UntrainedModelError


class TextSarcasmModel(nn.Module):
    def __init__(self, model_name: str = "roberta-base", dropout: float = 0.1) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = int(self.encoder.config.hidden_size)
        self.project = nn.Identity() if hidden == TEXT_HIDDEN else nn.Linear(hidden, TEXT_HIDDEN)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(TEXT_HIDDEN, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence = self.project(output.last_hidden_state)
        pooled = sequence[:, 0]
        logits = self.classifier(self.dropout(pooled))
        return {"logits": logits, "sequence_embeddings": sequence, "pooled": pooled}

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True


def save_text_checkpoint(
    path: str | Path,
    model: TextSarcasmModel,
    tokenizer,
    *,
    config: dict,
    epoch: int,
    metrics: dict,
    seed: int,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trained": True,
            "model_name": model.model_name,
            "model_state": model.state_dict(),
            "tokenizer_name": tokenizer.name_or_path,
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
            "seed": seed,
        },
        destination,
    )


def load_text_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple[TextSarcasmModel, object, dict]:
    source = Path(path)
    if not source.is_file():
        raise CheckpointNotFoundError(
            f"No text checkpoint at {source}. Train one with: python -m member1_text.train"
        )
    try:
        payload = torch.load(source, map_location=device or "cpu", weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location=device or "cpu")
    if not payload.get("trained"):
        raise UntrainedModelError(f"Checkpoint {source} is not marked as trained")
    from transformers import AutoTokenizer

    model = TextSarcasmModel(payload["model_name"])
    model.load_state_dict(payload["model_state"])
    model.to(device or "cpu")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(payload.get("tokenizer_name") or payload["model_name"])
    return model, tokenizer, payload
