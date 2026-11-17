"""Per-corpus loaders. Each corpus keeps its own schema and label meaning."""

from member1_text.data.mustard import load_mustard
from member1_text.data.mustard_pp import load_mustard_pp
from member1_text.data.sarc import load_sarc
from member1_text.data.semeval2018 import load_semeval

__all__ = ["load_mustard", "load_mustard_pp", "load_sarc", "load_semeval"]
