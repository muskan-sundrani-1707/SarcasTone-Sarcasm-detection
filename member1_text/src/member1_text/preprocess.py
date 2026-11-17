"""Unicode-safe preprocessing applied after a loader returns records.

Case, exclamation marks, question marks, ellipses, repeated letters, and
sarcasm markers such as ``/s`` are preserved. Contractions are recorded and
are expanded only when an ablation flag asks for it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from member1_text.canonicalize import CanonicalExample

_CURLY_QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
    }
)

# Fixed ablation dictionary. Matching is case-insensitive; expansion keeps
# the capitalization of the first character when the token starts a clause.
_CONTRACTIONS = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

_TOKEN = re.compile(r"\S+")


def normalize_text(text: str) -> str:
    """NFKC plus ASCII folding of curly apostrophes and quotes."""
    normalized = unicodedata.normalize("NFKC", text)
    return normalized.translate(_CURLY_QUOTES)


def _expand_token(token: str) -> str:
    match = re.match(r"^(\W*)(.*?)(\W*)$", token, flags=re.UNICODE)
    if match is None:
        return token
    prefix, core, suffix = match.groups()
    replacement = _CONTRACTIONS.get(core.casefold())
    if replacement is None:
        return token
    if core[:1].isupper():
        replacement = replacement[:1].upper() + replacement[1:]
    return f"{prefix}{replacement}{suffix}"


def contraction_tokens(text: str) -> list[str]:
    """Return contraction tokens found in already-normalized text."""
    found: list[str] = []
    for token in _TOKEN.findall(text):
        match = re.match(r"^(\W*)(.*?)(\W*)$", token, flags=re.UNICODE)
        core = match.group(2) if match else token
        if core.casefold() in _CONTRACTIONS:
            found.append(core)
    return found


def preprocess_text(text: str, *, expand_contractions: bool = False) -> tuple[str, dict[str, Any]]:
    """Normalize one string and optionally expand a fixed contraction list."""
    normalized = normalize_text(text)
    contractions = contraction_tokens(normalized)
    if expand_contractions:
        pieces: list[str] = []
        last = 0
        for match in _TOKEN.finditer(normalized):
            pieces.append(normalized[last : match.start()])
            pieces.append(_expand_token(match.group(0)))
            last = match.end()
        pieces.append(normalized[last:])
        normalized = "".join(pieces)
    return normalized, {
        "contractions": contractions,
        "expanded_contractions": expand_contractions,
        "original_text": text,
    }


def join_context(turns: list[dict[str, str]] | None) -> str | None:
    """Join context turns. A speaker is prefixed only when one is present."""
    if not turns:
        return None
    parts: list[str] = []
    for turn in turns:
        text = turn.get("text", "")
        speaker = turn.get("speaker") or ""
        if speaker:
            parts.append(f"{speaker}: {text}")
        else:
            parts.append(text)
    joined = " ".join(part for part in parts if part != "")
    return joined or None


def preprocess_example(
    example: CanonicalExample,
    *,
    expand_contractions: bool = False,
) -> CanonicalExample:
    """Return a copy with normalized target and context. The original text is kept."""
    processed, info = preprocess_text(example.text, expand_contractions=expand_contractions)
    new_turns: list[dict[str, str]] = []
    context_contractions: list[str] = []
    for turn in example.context_turns:
        turn_text, turn_info = preprocess_text(
            turn.get("text", ""),
            expand_contractions=expand_contractions,
        )
        updated = dict(turn)
        updated["text"] = turn_text
        if "original_text" not in updated:
            updated["original_text"] = turn.get("text", "")
        new_turns.append(updated)
        context_contractions.extend(turn_info["contractions"])
    extra = dict(example.extra)
    extra["original_text"] = example.text
    extra["contractions"] = info["contractions"]
    extra["context_contractions"] = context_contractions
    extra["expanded_contractions"] = expand_contractions
    extra["overlap_text"] = extra.get("overlap_text") or example.text
    return CanonicalExample(
        example_id=example.example_id,
        dataset=example.dataset,
        split=example.split,
        text=processed,
        context_turns=new_turns,
        text_source=example.text_source,
        label=example.label,
        label_meaning=example.label_meaning,
        extra=extra,
        overlap_key=example.overlap_key,
    )
