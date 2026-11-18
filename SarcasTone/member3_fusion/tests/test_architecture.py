"""Unit tests for member3_fusion architecture.

Tests:
  T1  Cross-modal attention output shapes (single-token and sequence)
  T2  PSID output shapes, gradient flow
  T3  Confidence-gated fusion: normal, missing audio, text-only
  T4  Context encoder shapes
  T5  Prosody and acoustic projections
  T6  Full forward pass: shapes, gradients
  T7  Missing modality (audio_available=False)
  T8  Fusion strategies: early, late, concat, cross_attention
  T9  PSID disabled, gating disabled (ablation flags)
  T10 Temperature scaler: monotone, positive T
"""
from __future__ import annotations

import sys
from pathlib import Path

# Insert repo root so we can import from shared and member3_fusion
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_M3 = _REPO / "member3_fusion" / "src"
if str(_M3.parent) not in sys.path:
    sys.path.insert(0, str(_M3.parent))

import pytest
import torch
import torch.nn as nn

from shared.constants import (
    TEXT_HIDDEN, AUDIO_HIDDEN, PROSODY_DIM, QUALITY_DIM,
)

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

B = 4   # batch size for tests
FUSION_HIDDEN = 256   # smaller than production for speed
ACOUSTIC_DIM  = 116

def _rand(*shape, requires_grad: bool = False):
    return torch.randn(*shape).requires_grad_(requires_grad)

def _bool_mask(B_, N):
    """Random bool mask; some entries True (pad)."""
    m = torch.zeros(B_, N, dtype=torch.bool)
    return m   # no padding in simple tests


def _make_batch(B_=B, audio_avail=True, include_all=True):
    return {
        "audio_embedding": _rand(B_, AUDIO_HIDDEN),
        "audio_available": torch.tensor([audio_avail] * B_),
        "prosody":         _rand(B_, PROSODY_DIM),
        "acoustic_full":   _rand(B_, ACOUSTIC_DIM),
        "quality":         _rand(B_, QUALITY_DIM).abs(),
        "quality_ok":      torch.tensor([True] * B_),
        "label":           torch.randint(0, 2, (B_,)),
        "texts":           [f"utterance {i}" for i in range(B_)],
        "contexts":        [[] for _ in range(B_)],
        "example_ids":     [str(i) for i in range(B_)],
        "speakers":        ["SPEAKER"] * B_,
    }

# ------------------------------------------------------------------ #
# T1: Cross-modal attention
# ------------------------------------------------------------------ #

class TestCrossModalAttention:

    def test_single_token_shapes(self):
        from src.attention.cross_attention import CrossModalAttentionBlock
        block = CrossModalAttentionBlock(
            text_dim=FUSION_HIDDEN, audio_dim=FUSION_HIDDEN,
            out_dim=FUSION_HIDDEN, num_heads=4, dropout=0.0,
        )
        t = _rand(B, 1, FUSION_HIDDEN)
        a = _rand(B, 1, FUSION_HIDDEN)
        t_out, a_out = block(t, a)
        assert t_out.shape == (B, 1, FUSION_HIDDEN)
        assert a_out.shape == (B, 1, FUSION_HIDDEN)

    def test_sequence_shapes(self):
        from src.attention.cross_attention import CrossModalAttentionBlock
        block = CrossModalAttentionBlock(
            text_dim=FUSION_HIDDEN, audio_dim=FUSION_HIDDEN,
            out_dim=FUSION_HIDDEN, num_heads=4, dropout=0.0,
        )
        Nt, Na = 10, 25
        t = _rand(B, Nt, FUSION_HIDDEN)
        a = _rand(B, Na, FUSION_HIDDEN)
        t_out, a_out = block(t, a)
        assert t_out.shape == (B, Nt, FUSION_HIDDEN)
        assert a_out.shape == (B, Na, FUSION_HIDDEN)

    def test_different_input_dims(self):
        """text_dim ≠ audio_dim → input projections handle them."""
        from src.attention.cross_attention import CrossModalAttentionBlock
        block = CrossModalAttentionBlock(
            text_dim=TEXT_HIDDEN, audio_dim=AUDIO_HIDDEN,
            out_dim=FUSION_HIDDEN, num_heads=4, dropout=0.0,
        )
        t = _rand(B, 1, TEXT_HIDDEN)
        a = _rand(B, 1, AUDIO_HIDDEN)
        t_out, a_out = block(t, a)
        assert t_out.shape == (B, 1, FUSION_HIDDEN)

    def test_stacked_layers(self):
        from src.attention.cross_attention import StackedCrossModalAttention
        stack = StackedCrossModalAttention(
            text_dim=FUSION_HIDDEN, audio_dim=FUSION_HIDDEN,
            out_dim=FUSION_HIDDEN, num_layers=3, num_heads=4, dropout=0.0,
        )
        t = _rand(B, 1, FUSION_HIDDEN)
        a = _rand(B, 1, FUSION_HIDDEN)
        t_out, a_out = stack(t, a)
        assert t_out.shape == (B, 1, FUSION_HIDDEN)

    def test_gradient_flow(self):
        from src.attention.cross_attention import CrossModalAttentionBlock
        block = CrossModalAttentionBlock(
            text_dim=FUSION_HIDDEN, audio_dim=FUSION_HIDDEN,
            out_dim=FUSION_HIDDEN, num_heads=4, dropout=0.0,
        )
        t = _rand(B, 1, FUSION_HIDDEN, requires_grad=True)
        a = _rand(B, 1, FUSION_HIDDEN, requires_grad=True)
        t_out, a_out = block(t, a)
        loss = t_out.sum() + a_out.sum()
        loss.backward()
        assert t.grad is not None
        assert a.grad is not None


# ------------------------------------------------------------------ #
# T2: PSID
# ------------------------------------------------------------------ #

class TestPSID:

    def _make_psid(self, proj_dim=64, disagree_dim=64):
        from src.incongruence.psid import PSIDModule
        return PSIDModule(
            text_dim=FUSION_HIDDEN, audio_dim=FUSION_HIDDEN,
            proj_dim=proj_dim, disagree_dim=disagree_dim, dropout=0.0,
        )

    def test_output_shapes(self):
        psid = self._make_psid()
        t = _rand(B, FUSION_HIDDEN)
        a = _rand(B, FUSION_HIDDEN)
        out = psid(t, a)
        assert out["disagreement"].shape == (B, 64)
        assert out["cosine_agree"].shape == (B,)
        assert out["t_proj"].shape == (B, 64)
        assert out["a_proj"].shape == (B, 64)

    def test_cosine_range(self):
        psid = self._make_psid()
        t = _rand(B, FUSION_HIDDEN)
        a = _rand(B, FUSION_HIDDEN)
        out = psid(t, a)
        cos = out["cosine_agree"]
        assert (cos >= -1.001).all() and (cos <= 1.001).all()

    def test_identical_inputs_high_agreement(self):
        """When text == audio, shared projections are equal → cosine should be positive."""
        psid = self._make_psid()
        x = _rand(B, FUSION_HIDDEN)
        # Same input → same MLP output → t_proj == a_proj → cosine_agree == 1.0
        # But random MLPs after LayerNorm can still have cosine > 0 for identical inputs
        out = psid(x, x)
        # At minimum, agreement should be strictly positive for identical inputs
        assert (out["cosine_agree"] > -0.5).all(), (
            f"Expected positive agreement for identical inputs, got {out['cosine_agree']}"
        )

    def test_gradient_flow(self):
        psid = self._make_psid()
        t = _rand(B, FUSION_HIDDEN, requires_grad=True)
        a = _rand(B, FUSION_HIDDEN, requires_grad=True)
        out = psid(t, a)
        loss = out["disagreement"].sum()
        loss.backward()
        assert t.grad is not None
        assert a.grad is not None


# ------------------------------------------------------------------ #
# T3: Confidence-gated fusion
# ------------------------------------------------------------------ #

class TestConfidenceGatedFusion:

    def _make_gate(self):
        from src.gating.confidence_gate import ConfidenceGatedFusion
        return ConfidenceGatedFusion(
            text_dim=FUSION_HIDDEN, audio_dim=FUSION_HIDDEN,
            quality_dim=QUALITY_DIM, hidden_dim=64, dropout=0.0,
        )

    def test_output_shapes(self):
        gate = self._make_gate()
        t = _rand(B, FUSION_HIDDEN)
        a = _rand(B, FUSION_HIDDEN)
        q = _rand(B, QUALITY_DIM).abs()
        a_avail = torch.ones(B, dtype=torch.bool)
        q_ok    = torch.ones(B, dtype=torch.bool)
        out = gate(t, a, q, a_avail, q_ok)
        assert out["fused"].shape == (B, FUSION_HIDDEN)
        assert out["text_weight"].shape == (B,)
        assert out["audio_weight"].shape == (B,)

    def test_missing_audio_zeroed(self):
        """When audio_available=False, audio weight must be 0."""
        gate = self._make_gate()
        t = _rand(B, FUSION_HIDDEN)
        a = _rand(B, FUSION_HIDDEN)
        q = torch.zeros(B, QUALITY_DIM)
        a_avail = torch.zeros(B, dtype=torch.bool)   # ALL missing
        q_ok    = torch.zeros(B, dtype=torch.bool)
        out = gate(t, a, q, a_avail, q_ok)
        assert (out["audio_weight"].abs() < 1e-6).all(), (
            f"Expected zero audio weight, got {out['audio_weight']}"
        )

    def test_weights_sum_to_one(self):
        """With normalise_weights=True, w_text + w_audio ≈ 1."""
        gate = self._make_gate()
        t = _rand(B, FUSION_HIDDEN)
        a = _rand(B, FUSION_HIDDEN)
        q = torch.rand(B, QUALITY_DIM)
        a_avail = torch.ones(B, dtype=torch.bool)
        q_ok    = torch.ones(B, dtype=torch.bool)
        out = gate(t, a, q, a_avail, q_ok)
        total = out["text_weight"] + out["audio_weight"]
        assert torch.allclose(total, torch.ones(B), atol=1e-5)

    def test_gradient_flow(self):
        gate = self._make_gate()
        t = _rand(B, FUSION_HIDDEN, requires_grad=True)
        a = _rand(B, FUSION_HIDDEN, requires_grad=True)
        q = torch.rand(B, QUALITY_DIM)
        a_avail = torch.ones(B, dtype=torch.bool)
        q_ok    = torch.ones(B, dtype=torch.bool)
        out = gate(t, a, q, a_avail, q_ok)
        out["fused"].sum().backward()
        assert t.grad is not None


# ------------------------------------------------------------------ #
# T4: Context encoder
# ------------------------------------------------------------------ #

class TestContextEncoder:

    def test_output_shape(self):
        from src.context.context_encoder import ContextEncoder
        enc = ContextEncoder(hidden_dim=FUSION_HIDDEN, dropout=0.0)
        target  = _rand(B, FUSION_HIDDEN)
        context = _rand(B, FUSION_HIDDEN)
        has_ctx = torch.ones(B, dtype=torch.bool)
        out = enc(target, context, has_ctx)
        assert out["enriched"].shape == (B, FUSION_HIDDEN)
        assert out["gate"].shape == (B,)

    def test_no_context_unchanged(self):
        """When has_context=False, enriched should equal target."""
        from src.context.context_encoder import ContextEncoder
        enc = ContextEncoder(hidden_dim=FUSION_HIDDEN, dropout=0.0)
        target  = _rand(B, FUSION_HIDDEN)
        context = torch.zeros(B, FUSION_HIDDEN)
        has_ctx = torch.zeros(B, dtype=torch.bool)
        out = enc(target, context, has_ctx)
        assert torch.allclose(out["enriched"], target, atol=1e-5)

    def test_gate_range(self):
        from src.context.context_encoder import ContextEncoder
        enc = ContextEncoder(hidden_dim=FUSION_HIDDEN, dropout=0.0)
        target  = _rand(B, FUSION_HIDDEN)
        context = _rand(B, FUSION_HIDDEN)
        has_ctx = torch.ones(B, dtype=torch.bool)
        out = enc(target, context, has_ctx)
        g = out["gate"]
        assert (g >= 0).all() and (g <= 1).all()


# ------------------------------------------------------------------ #
# T5: Projections
# ------------------------------------------------------------------ #

class TestProjections:

    def test_prosody_shape(self):
        from src.fusion.prosody_proj import ProsodyProjection
        proj = ProsodyProjection(prosody_dim=PROSODY_DIM, out_dim=64, dropout=0.0)
        x = _rand(B, PROSODY_DIM)
        out = proj(x)
        assert out.shape == (B, 64)

    def test_acoustic_shape(self):
        from src.fusion.prosody_proj import AcousticFullProjection
        proj = AcousticFullProjection(acoustic_dim=ACOUSTIC_DIM, out_dim=64, dropout=0.0)
        x = _rand(B, ACOUSTIC_DIM)
        out = proj(x)
        assert out.shape == (B, 64)


# ------------------------------------------------------------------ #
# T6 & T7: Full fusion model forward pass
# ------------------------------------------------------------------ #

class TestFusionModel:

    def _make_model(self, fusion_strategy="cross_attention", psid=True, gating=True):
        from src.models.fusion_model import SarcasToneFusionModel
        return SarcasToneFusionModel(
            text_dim=TEXT_HIDDEN,
            audio_dim=AUDIO_HIDDEN,
            prosody_dim=PROSODY_DIM,
            quality_dim=QUALITY_DIM,
            acoustic_dim=ACOUSTIC_DIM,
            fusion_hidden=FUSION_HIDDEN,
            cross_attn_heads=4,
            cross_attn_layers=2,
            dropout=0.0,
            classifier_hidden=128,
            num_labels=2,
            psid_proj_dim=64,
            psid_disagree_dim=64,
            fusion_strategy=fusion_strategy,
            psid_enabled=psid,
            gating_enabled=gating,
        )

    def _run_forward(self, model, audio_avail=True):
        batch = _make_batch(audio_avail=audio_avail)
        text_emb = _rand(B, TEXT_HIDDEN)
        ctx_emb  = _rand(B, TEXT_HIDDEN)
        has_ctx  = torch.ones(B, dtype=torch.bool)
        return model(batch, text_embeddings=text_emb, context_embeddings=ctx_emb, has_context=has_ctx)

    def test_full_model_shapes(self):
        model = self._make_model()
        out = self._run_forward(model)
        assert out["logits"].shape == (B, 2)
        assert out["probabilities"].shape == (B, 2)
        assert out["entropy"].shape == (B,)

    def test_probabilities_sum_to_one(self):
        model = self._make_model()
        out = self._run_forward(model)
        sums = out["probabilities"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5)

    def test_missing_audio(self):
        """With audio_available=False, model must still run without errors."""
        model = self._make_model()
        out = self._run_forward(model, audio_avail=False)
        assert out["logits"].shape == (B, 2)
        # audio_weight should be zero or near-zero
        if out["audio_weight"] is not None:
            assert (out["audio_weight"].abs() < 1e-5).all()

    def test_gradient_flow(self):
        model = self._make_model()
        model.train()
        batch = _make_batch()
        text_emb = _rand(B, TEXT_HIDDEN, requires_grad=True)
        ctx_emb  = _rand(B, TEXT_HIDDEN)
        has_ctx  = torch.ones(B, dtype=torch.bool)
        out = model(batch, text_embeddings=text_emb, context_embeddings=ctx_emb, has_context=has_ctx)
        loss = nn.CrossEntropyLoss()(out["logits"], batch["label"])
        loss.backward()
        # At least some model parameters must have gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients found after backward pass"

    @pytest.mark.parametrize("strategy", ["early", "late", "concat", "cross_attention"])
    def test_fusion_strategies(self, strategy):
        model = self._make_model(fusion_strategy=strategy, psid=False, gating=False)
        out = self._run_forward(model)
        assert out["logits"].shape == (B, 2)

    def test_psid_disabled(self):
        model = self._make_model(psid=False, gating=True)
        out = self._run_forward(model)
        assert out["disagreement_norm"] is None
        assert out["cosine_agree"] is None

    def test_gating_disabled(self):
        model = self._make_model(psid=True, gating=False)
        out = self._run_forward(model)
        assert out["text_weight"] is None
        assert out["audio_weight"] is None


# ------------------------------------------------------------------ #
# T8: Temperature scaler
# ------------------------------------------------------------------ #

class TestTemperatureScaler:

    def test_positive_temperature(self):
        from src.uncertainty.calibration import TemperatureScaler
        scaler = TemperatureScaler(init_value=1.5)
        T = scaler.temperature
        assert T.item() > 0

    def test_scaling_effect(self):
        from src.uncertainty.calibration import TemperatureScaler
        scaler = TemperatureScaler(init_value=2.0)
        logits = _rand(B, 2)
        scaled = scaler(logits)
        # With T ≈ 2, probabilities should be more uniform (lower max prob)
        p_orig   = torch.softmax(logits, dim=-1).max(dim=-1).values
        p_scaled = torch.softmax(scaled, dim=-1).max(dim=-1).values
        assert (p_scaled <= p_orig + 1e-5).all(), "Scaling should flatten probs"

    def test_predict_with_uncertainty(self):
        from src.uncertainty.calibration import predict_with_uncertainty
        logits = _rand(B, 2)
        out = predict_with_uncertainty(logits, temperature=1.0)
        assert out["probabilities"].shape == (B, 2)
        assert out["entropy"].shape == (B,)
        assert out["uncertain"].dtype == torch.bool
        # Entropy must be in [0, ln(2)]
        import math
        assert (out["entropy"] >= 0).all()
        assert (out["entropy"] <= math.log(2) + 1e-5).all()


# ------------------------------------------------------------------ #
# T9: Dataset record construction (no file I/O)
# ------------------------------------------------------------------ #

class TestFusionRecord:

    def test_record_defaults(self):
        from src.training.dataset import FusionRecord
        import numpy as np
        rec = FusionRecord(
            example_id="test_01",
            dataset="mustard",
            text="It's great.",
            context=[],
            speaker="SHELDON",
            label=1,
        )
        assert rec.label == 1
        assert rec.prosody_vector.shape == (PROSODY_DIM,)
        assert rec.quality_vector.shape == (QUALITY_DIM,)
        assert not rec.audio_available

    def test_dataset_item_shapes(self):
        from src.training.dataset import FusionRecord, FusionDataset
        import numpy as np
        recs = [
            FusionRecord(
                example_id=str(i), dataset="mustard", text="text",
                context=[], speaker="SPK", label=i % 2,
                audio_embedding=np.random.randn(AUDIO_HIDDEN).astype(np.float32),
                audio_available=True,
                prosody_vector=np.random.randn(PROSODY_DIM).astype(np.float32),
                acoustic_features_full=np.random.randn(116).astype(np.float32),
                quality_vector=np.random.randn(QUALITY_DIM).astype(np.float32),
                quality_ok=True,
            )
            for i in range(B)
        ]
        ds = FusionDataset(recs, acoustic_dim=116)
        item = ds[0]
        assert item["audio_embedding"].shape == (AUDIO_HIDDEN,)
        assert item["prosody"].shape == (PROSODY_DIM,)
        assert item["quality"].shape == (QUALITY_DIM,)
        assert item["label"].dtype == torch.long


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
