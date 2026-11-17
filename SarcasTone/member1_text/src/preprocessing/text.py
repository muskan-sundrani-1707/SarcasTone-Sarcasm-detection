"""Text normalization.

Capitalization and punctuation stay in the string because they are sarcasm
cues ("Oh, GREAT..."). Contraction expansion is available and off by default:
RoBERTa and DeBERTa were pretrained on contractions, so expanding them is a
separate experiment rather than a silent preprocessing step.
"""

from __future__ import annotations

import re
import unicodedata

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
}

_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]")
_WHITESPACE = re.compile(r"[ \t]+")


def normalize_text(text: str, *, expand_contractions: bool = False) -> str:
    if text is None:
        return ""
    cleaned = unicodedata.normalize("NFKC", str(text))
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL.sub("", cleaned)
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    if expand_contractions:
        cleaned = expand_contraction_text(cleaned)
    return cleaned


def expand_contraction_text(text: str) -> str:
    """Expand a small set of English contractions. Case of the stem is kept."""

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        lower = token.lower()
        if lower in {"can't", "won't"}:
            replacement = CONTRACTIONS[lower]
            return replacement if token[0].islower() else replacement.capitalize()
        return token

    text = re.sub(r"\b(?:can't|won't)\b", _replace, text, flags=re.IGNORECASE)

    def _suffix(match: re.Match[str]) -> str:
        stem, suffix = match.group(1), match.group(2).lower()
        expansion = CONTRACTIONS.get(suffix)
        if expansion is None:
            return match.group(0)
        return stem + expansion

    return re.sub(r"\b([A-Za-z]+)(n't|'re|'ve|'ll|'d|'m)\b", _suffix, text)


def normalize_context(turns: list[str] | None, *, expand_contractions: bool = False) -> list[str]:
    if not turns:
        return []
    cleaned = []
    for turn in turns:
        text = normalize_text(turn, expand_contractions=expand_contractions)
        if text:
            cleaned.append(text)
    return cleaned
