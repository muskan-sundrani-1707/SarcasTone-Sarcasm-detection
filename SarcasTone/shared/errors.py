"""Exceptions raised by more than one module."""


class ArcasToneError(Exception):
    """Base error for expected, user-facing failures."""


class DatasetSchemaError(ArcasToneError):
    """A dataset file does not match the schema this loader understands."""


class CheckpointNotFoundError(ArcasToneError, FileNotFoundError):
    """A required trained checkpoint is not on disk."""


class UntrainedModelError(ArcasToneError):
    """Inference was asked to treat an untrained head as a sarcasm model."""


class AudioLoadError(ArcasToneError):
    """Audio could not be decoded, was empty, or failed a duration check."""
