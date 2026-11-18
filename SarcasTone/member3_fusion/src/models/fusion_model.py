"""SarcasTone Fusion Model — central multimodal architecture.

Architecture (forward path)
----------------------------
Input batch keys:
  texts             list[str]           B utterances
  contexts          list[list[str]]     B context turn lists
  audio_embedding   [B, AUDIO_HIDDEN]   pre-extracted wav2vec2 pooled
  audio_available   [B] bool
  prosody           [B, PROSODY_DIM]    handcrafted prosody features
  acoustic_full     [B, acoustic_dim]  all 116 handoff columns
  quality           [B, QUALITY_DIM]   audio quality signals
  quality_ok        [B] bool

Forward stages:
  1.  Text encoder (member1 SpanPoolingModel or identity)
      → text_pool [B, 768], ctx_pool [B, 768], has_ctx [B]

  2.  Context encoder
      → text_enriched [B, 768]

  3.  Prosody projection
      → prosody_proj [B, psid_proj_dim]

  4.  Acoustic-full projection
      → acoustic_proj [B, fusion_hidden//4]

  5.  Cross-modal attention (text ↔ audio)
      → text_xattn [B, fusion_hidden], audio_xattn [B, fusion_hidden]

  6.  PSID (optional)
      → disagreement [B, disagree_dim]

  7.  Confidence-gated fusion (optional)
      → fused [B, fusion_hidden], text_weight [B], audio_weight [B]
         OR simple concatenation fallback

  8.  Classifier input assembly
      concat( fused, disagreement, acoustic_proj )
      → [B, classifier_in_dim]

  9.  Residual projection + LayerNorm + Dropout

  10. Classification head → logits [B, 2]

  11. Temperature scaler (post-training)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
_REPO = Path(__file__).resolve().parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from shared.constants import (
    TEXT_HIDDEN, AUDIO_HIDDEN, CONTEXT_HIDDEN,
    PROSODY_DIM, QUALITY_DIM, FUSION_LOGITS,
)

from member3_fusion.src.attention.cross_attention import StackedCrossModalAttention
from member3_fusion.src.incongruence.psid import PSIDModule
from member3_fusion.src.gating.confidence_gate import ConfidenceGatedFusion
from member3_fusion.src.context.context_encoder import ContextEncoder
from member3_fusion.src.fusion.prosody_proj import ProsodyProjection, AcousticFullProjection
from member3_fusion.src.uncertainty.calibration import TemperatureScaler

log = logging.getLogger(__name__)


class SarcasToneFusionModel(nn.Module):
    """Central multimodal sarcasm detection model.

    Fusion strategy controlled by ``fusion_strategy``:
        "early"          – concatenate raw embeddings before attention
        "late"           – average independent predictions
        "concat"         – concatenate cross-attended vectors
        "cross_attention"– full bidirectional cross-attention + gating

    PSID and gating are optional sub-modules; set their flags in the config.
    """

    FUSION_STRATEGIES = ("early", "late", "concat", "cross_attention")

    def __init__(
        self,
        # Upstream widths
        text_dim: int = TEXT_HIDDEN,
        audio_dim: int = AUDIO_HIDDEN,
        context_dim: int = CONTEXT_HIDDEN,
        prosody_dim: int = PROSODY_DIM,
        quality_dim: int = QUALITY_DIM,
        acoustic_dim: int = 116,
        # Model hyper-params
        fusion_hidden: int = 512,
        cross_attn_heads: int = 8,
        cross_attn_layers: int = 2,
        dropout: float = 0.3,
        classifier_hidden: int = 256,
        num_labels: int = 2,
        psid_proj_dim: int = 128,
        psid_disagree_dim: int = 128,
        temperature_init: float = 1.0,
        # Feature flags
        fusion_strategy: str = "cross_attention",
        psid_enabled: bool = True,
        gating_enabled: bool = True,
        text_available: bool = True,
        audio_available_default: bool = True,
    ) -> None:
        super().__init__()

        if fusion_strategy not in self.FUSION_STRATEGIES:
            raise ValueError(
                f"fusion_strategy must be one of {self.FUSION_STRATEGIES}, "
                f"got '{fusion_strategy}'"
            )

        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.prosody_dim = prosody_dim
        self.quality_dim = quality_dim
        self.acoustic_dim = acoustic_dim
        self.fusion_hidden = fusion_hidden
        self.num_labels = num_labels
        self.fusion_strategy = fusion_strategy
        self.psid_enabled = psid_enabled
        self.gating_enabled = gating_enabled
        self.text_available_flag = text_available
        self.audio_available_default = audio_available_default

        # ------------------------------------------------------------------ #
        # 1. Input projection layers (bring encoder outputs to fusion_hidden)
        # ------------------------------------------------------------------ #
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------ #
        # 2. Context encoder
        # ------------------------------------------------------------------ #
        self.context_encoder = ContextEncoder(
            hidden_dim=text_dim, dropout=dropout
        )

        # ------------------------------------------------------------------ #
        # 3. Prosody and acoustic projections
        # ------------------------------------------------------------------ #
        prosody_out = psid_proj_dim
        self.prosody_proj = ProsodyProjection(
            prosody_dim=prosody_dim, out_dim=prosody_out, dropout=dropout
        )

        acou_out = fusion_hidden // 4   # e.g. 128 for fusion_hidden=512
        self.acoustic_full_proj = AcousticFullProjection(
            acoustic_dim=acoustic_dim, out_dim=acou_out, dropout=dropout
        )

        # ------------------------------------------------------------------ #
        # 4 & 5. Cross-modal attention
        # ------------------------------------------------------------------ #
        if fusion_strategy in ("cross_attention", "concat"):
            self.cross_attn = StackedCrossModalAttention(
                text_dim=fusion_hidden,
                audio_dim=fusion_hidden,
                out_dim=fusion_hidden,
                num_layers=cross_attn_layers,
                num_heads=cross_attn_heads,
                dropout=dropout,
            )
        else:
            self.cross_attn = None

        # ------------------------------------------------------------------ #
        # 6. PSID module
        # ------------------------------------------------------------------ #
        if psid_enabled:
            # PSID operates on the cross-attended (or projected) representations
            self.psid = PSIDModule(
                text_dim=fusion_hidden,
                audio_dim=fusion_hidden,
                proj_dim=psid_proj_dim,
                disagree_dim=psid_disagree_dim,
                dropout=dropout,
            )
            disagree_in = psid_disagree_dim
        else:
            self.psid = None
            disagree_in = 0

        # ------------------------------------------------------------------ #
        # 7. Confidence-gated fusion
        # ------------------------------------------------------------------ #
        if gating_enabled:
            self.conf_gate = ConfidenceGatedFusion(
                text_dim=fusion_hidden,
                audio_dim=fusion_hidden,
                quality_dim=quality_dim,
                hidden_dim=classifier_hidden,
                dropout=dropout,
            )
        else:
            self.conf_gate = None

        # ------------------------------------------------------------------ #
        # 8. Compute classifier input dimension
        # ------------------------------------------------------------------ #
        # fused vector width depends on strategy
        if fusion_strategy in ("early",):
            fused_width = text_dim + audio_dim
            # extra projection to fusion_hidden
            self.early_proj = nn.Linear(fused_width, fusion_hidden)
            classifier_in_dim = fusion_hidden
        elif fusion_strategy == "late":
            # Text and audio classifiers individually; combined via averaging
            # We still build one common head; each modality is projected
            classifier_in_dim = fusion_hidden
        elif fusion_strategy == "concat":
            classifier_in_dim = fusion_hidden * 2  # [text_xattn; audio_xattn]
        else:  # cross_attention
            classifier_in_dim = fusion_hidden       # gated or averaged

        # Add disagreement and acoustic-full vectors
        classifier_in_dim += disagree_in + acou_out

        # ------------------------------------------------------------------ #
        # 9. Residual projection + normalisation
        # ------------------------------------------------------------------ #
        self.res_proj = nn.Sequential(
            nn.Linear(classifier_in_dim, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------ #
        # 10. Classification head
        # ------------------------------------------------------------------ #
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_labels),
        )

        # ------------------------------------------------------------------ #
        # 11. Temperature scaler (post-training)
        # ------------------------------------------------------------------ #
        self.temperature_scaler = TemperatureScaler(init_value=temperature_init)

        self._init_weights()

        # Store config for serialisation
        self._config = {
            "text_dim": text_dim,
            "audio_dim": audio_dim,
            "context_dim": context_dim,
            "prosody_dim": prosody_dim,
            "quality_dim": quality_dim,
            "acoustic_dim": acoustic_dim,
            "fusion_hidden": fusion_hidden,
            "cross_attn_heads": cross_attn_heads,
            "cross_attn_layers": cross_attn_layers,
            "dropout": dropout,
            "classifier_hidden": classifier_hidden,
            "num_labels": num_labels,
            "psid_proj_dim": psid_proj_dim,
            "psid_disagree_dim": psid_disagree_dim,
            "temperature_init": temperature_init,
            "fusion_strategy": fusion_strategy,
            "psid_enabled": psid_enabled,
            "gating_enabled": gating_enabled,
        }

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------------- #
    # Forward pass
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        batch: dict,
        *,
        text_embeddings: Optional[torch.Tensor] = None,  # [B, text_dim]  pre-computed
        context_embeddings: Optional[torch.Tensor] = None,  # [B, ctx_dim]
        has_context: Optional[torch.Tensor] = None,         # [B] bool
        return_logits_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            batch:  dict from collate_fn containing tensors and lists.
            text_embeddings: pre-computed text pool (skips text encoder call).
            context_embeddings: pre-computed context pool.
            has_context: bool mask.
            return_logits_only: return only logits (used by temperature fitting).

        Returns dict containing:
            logits:        [B, 2]
            probabilities: [B, 2]
            text_weight:   [B] (or None)
            audio_weight:  [B] (or None)
            disagreement_norm: [B] (or None)
            cosine_agree:  [B] (or None)
            entropy:       [B]
        """
        device = next(self.parameters()).device

        audio_emb_raw   = batch["audio_embedding"].to(device)    # [B, AUDIO_HIDDEN]
        audio_avail     = batch["audio_available"].to(device)    # [B] bool
        prosody         = batch["prosody"].to(device)             # [B, PROSODY_DIM]
        acoustic_full_v = batch["acoustic_full"].to(device)      # [B, acoustic_dim]
        quality         = batch["quality"].to(device)             # [B, QUALITY_DIM]
        quality_ok      = batch["quality_ok"].to(device)          # [B] bool
        B = audio_emb_raw.size(0)

        # ------------------------------------------------------------------ #
        # 1. Text representations
        # ------------------------------------------------------------------ #
        if text_embeddings is not None:
            text_pool = text_embeddings.to(device)      # [B, text_dim]
        else:
            # Zero text if no encoder (text-unavailable ablation)
            text_pool = torch.zeros(B, self.text_dim, device=device)

        if context_embeddings is not None:
            ctx_pool = context_embeddings.to(device)
            hc = has_context.to(device) if has_context is not None else torch.zeros(B, dtype=torch.bool, device=device)
        else:
            ctx_pool = torch.zeros(B, self.text_dim, device=device)
            hc = torch.zeros(B, dtype=torch.bool, device=device)

        # ------------------------------------------------------------------ #
        # 2. Context enrichment
        # ------------------------------------------------------------------ #
        ctx_out = self.context_encoder(text_pool, ctx_pool, hc)
        text_enriched = ctx_out["enriched"]   # [B, text_dim]

        # ------------------------------------------------------------------ #
        # 3. Prosody and acoustic projections
        # ------------------------------------------------------------------ #
        prosody_proj_out = self.prosody_proj(prosody)              # [B, psid_proj_dim]
        acoustic_proj_out = self.acoustic_full_proj(acoustic_full_v)  # [B, acou_out]

        # Mask acoustic outputs where audio is unavailable
        a_mask = audio_avail.float().unsqueeze(-1)
        acoustic_proj_out = acoustic_proj_out * a_mask
        prosody_proj_out  = prosody_proj_out  * a_mask

        # ------------------------------------------------------------------ #
        # 4 & 5. Project text and audio to fusion_hidden
        # ------------------------------------------------------------------ #
        text_h  = self.text_proj(text_enriched)      # [B, fusion_hidden]
        audio_h = self.audio_proj(audio_emb_raw)     # [B, fusion_hidden]
        audio_h = audio_h * a_mask                   # zero-out if unavailable

        # ------------------------------------------------------------------ #
        # Fusion strategy
        # ------------------------------------------------------------------ #
        text_weight  = None
        audio_weight = None

        if self.fusion_strategy == "early":
            # Concatenate raw, then project
            combined = torch.cat([text_enriched, audio_emb_raw * a_mask], dim=-1)
            fused = self.early_proj(combined)   # [B, fusion_hidden]
            text_cross  = fused
            audio_cross = fused

        elif self.fusion_strategy == "late":
            # Average projected representations (simple late fusion)
            fused = (text_h + audio_h) / 2.0
            text_cross  = text_h
            audio_cross = audio_h

        elif self.fusion_strategy == "concat":
            # Cross-attended then concatenated
            t_seq = text_h.unsqueeze(1)    # [B, 1, fusion_hidden]
            a_seq = audio_h.unsqueeze(1)   # [B, 1, fusion_hidden]
            t_out, a_out = self.cross_attn(t_seq, a_seq)
            text_cross  = t_out.squeeze(1)   # [B, fusion_hidden]
            audio_cross = a_out.squeeze(1)   # [B, fusion_hidden]
            fused = torch.cat([text_cross, audio_cross], dim=-1)  # [B, 2*FH]

        else:  # cross_attention
            t_seq = text_h.unsqueeze(1)
            a_seq = audio_h.unsqueeze(1)
            t_out, a_out = self.cross_attn(t_seq, a_seq)
            text_cross  = t_out.squeeze(1)
            audio_cross = a_out.squeeze(1)

            if self.gating_enabled and self.conf_gate is not None:
                gate_out = self.conf_gate(
                    text_emb=text_cross,
                    audio_emb=audio_cross,
                    quality_vec=quality,
                    audio_available=audio_avail,
                    quality_ok=quality_ok,
                )
                fused        = gate_out["fused"]
                text_weight  = gate_out["text_weight"]
                audio_weight = gate_out["audio_weight"]
            else:
                fused = (text_cross + audio_cross) / 2.0

        # ------------------------------------------------------------------ #
        # 6. PSID
        # ------------------------------------------------------------------ #
        disagreement_norm = None
        cosine_agree      = None
        disagree_vec      = torch.zeros(0, device=device)   # placeholder

        if self.psid_enabled and self.psid is not None:
            psid_out = self.psid(text_cross, audio_cross)
            disagree_vec      = psid_out["disagreement"]   # [B, disagree_dim]
            cosine_agree      = psid_out["cosine_agree"]   # [B]
            disagreement_norm = disagree_vec.norm(dim=-1)  # [B]

        # ------------------------------------------------------------------ #
        # 8. Assemble classifier input
        # ------------------------------------------------------------------ #
        parts = [fused, acoustic_proj_out]
        if self.psid_enabled and disagree_vec.numel() > 0:
            parts.append(disagree_vec)
        classifier_input = torch.cat(parts, dim=-1)   # [B, classifier_in_dim]

        # ------------------------------------------------------------------ #
        # 9. Residual projection + normalisation
        # ------------------------------------------------------------------ #
        hidden = self.res_proj(classifier_input)   # [B, classifier_hidden]

        # ------------------------------------------------------------------ #
        # 10. Classification head
        # ------------------------------------------------------------------ #
        logits = self.classifier(hidden)   # [B, 2]

        if return_logits_only:
            return {"logits": logits}

        # ------------------------------------------------------------------ #
        # 11. Temperature scaling (at inference; T=1 during training)
        # ------------------------------------------------------------------ #
        # During training we don't scale (T fixed at 1 via init)
        probs = torch.softmax(logits, dim=-1)
        from member3_fusion.src.uncertainty.calibration import predictive_entropy
        entropy = predictive_entropy(probs)

        return {
            "logits": logits,
            "probabilities": probs,
            "text_weight": text_weight,
            "audio_weight": audio_weight,
            "disagreement_norm": disagreement_norm,
            "cosine_agree": cosine_agree,
            "entropy": entropy,
            "context_gate": ctx_out["gate"],
        }

    # ---------------------------------------------------------------------- #
    # Serialisation
    # ---------------------------------------------------------------------- #

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "model.pt")
        (directory / "config.json").write_text(
            json.dumps(self._config, indent=2) + "\n", encoding="utf-8"
        )
        log.info("Saved fusion model to %s", directory)

    @classmethod
    def load(cls, directory: Path | str, device: torch.device | None = None) -> "SarcasToneFusionModel":
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        model = cls(**config)
        state = torch.load(directory / "model.pt", map_location=device or "cpu")
        model.load_state_dict(state)
        if device:
            model.to(device)
        model.eval()
        log.info("Loaded fusion model from %s", directory)
        return model


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_fusion_model(cfg: dict) -> SarcasToneFusionModel:
    """Build a SarcasToneFusionModel from a config dict."""
    m = cfg.get("model", {})
    exp = {}  # experiment-specific overrides already merged by load_config

    return SarcasToneFusionModel(
        text_dim=cfg["interface"]["text_hidden"],
        audio_dim=cfg["interface"]["audio_hidden"],
        context_dim=cfg["interface"]["context_hidden"],
        prosody_dim=cfg["interface"]["prosody_dim"],
        quality_dim=cfg["interface"]["quality_dim"],
        acoustic_dim=cfg["interface"].get("acoustic_feature_dim", 116),
        fusion_hidden=m.get("fusion_hidden", 512),
        cross_attn_heads=m.get("cross_attn_heads", 8),
        cross_attn_layers=m.get("cross_attn_layers", 2),
        dropout=m.get("dropout", 0.3),
        classifier_hidden=m.get("classifier_hidden", 256),
        num_labels=m.get("num_labels", 2),
        psid_proj_dim=m.get("psid_proj_dim", 128),
        psid_disagree_dim=m.get("psid_disagreement_dim", 128),
        temperature_init=m.get("temperature_init", 1.0),
        fusion_strategy=m.get("fusion_strategy", "cross_attention"),
        psid_enabled=m.get("psid_enabled", True),
        gating_enabled=m.get("gating_enabled", True),
        text_available=m.get("text_available", True),
        audio_available_default=m.get("audio_available", True),
    )
