"""Token-level Integrated Gradients for a trained text checkpoint.

Attribution is not a causal explanation. The returned scores say how much
each token changed the sarcastic-class logit relative to a pad baseline
under the Integrated Gradients path.
"""

from __future__ import annotations

import torch

from member1_text.src.models.text_encoder import TextSarcasmModel


def attribute_tokens(
    model: TextSarcasmModel,
    tokenizer,
    text: str,
    context: list[str] | None = None,
    *,
    max_length: int = 128,
    n_steps: int = 32,
) -> dict:
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise ImportError("Install captum to compute Integrated Gradients") from exc

    model.eval()
    device = next(model.parameters()).device
    if context:
        context_text = " ".join(turn for turn in context if turn.strip())
        encoded = tokenizer(
            context_text,
            text,
            truncation="only_first",
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
    else:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    embeddings = model.encoder.get_input_embeddings()(input_ids)
    baseline = torch.zeros_like(embeddings)

    def _logits_from_embeds(embeds: torch.Tensor) -> torch.Tensor:
        output = model.encoder(inputs_embeds=embeds, attention_mask=attention_mask)
        sequence = model.project(output.last_hidden_state)
        return model.classifier(sequence[:, 0])[:, 1]

    attributions = IntegratedGradients(_logits_from_embeds).attribute(
        embeddings, baselines=baseline, n_steps=n_steps
    )
    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu())
    mask = attention_mask.squeeze(0).detach().cpu().bool()
    rows = []
    for token, score, keep in zip(tokens, token_scores.tolist(), mask.tolist()):
        if not keep:
            continue
        rows.append({"token": token, "score": float(score)})
    return {
        "tokens": rows,
        "target_class": "sarcastic",
        "method": "integrated_gradients",
        "note": "Scores are attributions relative to a zero embedding baseline, not causal effects.",
    }
