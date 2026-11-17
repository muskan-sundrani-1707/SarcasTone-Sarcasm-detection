# SarcasTone

SarcasTone is a research project for sarcasm detection in speech, combining
textual context with acoustic and prosodic information.

## Current implementation

- `SarcasTone/shared/` defines common labels, dataset records, errors, tensor
  dimensions, and the interface contract.
- `SarcasTone/member1_text/` contains text normalization, dataset loaders,
  text encoding, optional speech transcription, and token attribution.
- `SarcasTone/configs/interface_contract.yaml` records the shared model shapes
  and conventions. The other member directories are currently placeholders;
  audio feature extraction, multimodal fusion, and deployment are not yet
  implemented in this project folder.

## Setup

Use Python 3.11 or newer. From the `SarcasTone` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

See [`SarcasTone/datasets/README.md`](SarcasTone/datasets/README.md) for dataset
formats, acquisition commands, and licensing notes. Raw datasets and trained
checkpoints are intentionally excluded from version control.
