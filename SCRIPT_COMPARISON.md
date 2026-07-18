# Script Comparison: Main Repository Scripts vs `multimodal/`

## Executive Summary

This repository currently contains a functional **single-modality (text-only) scripting pipeline** in the project root and `scripts/` directory, while the `multimodal/` directory is currently a **placeholder** with only a README link and no executable scripts.

As a result:
- The main pipeline can preprocess, sample, validate, and train sarcasm classifiers.
- The multimodal area does not yet provide runnable preprocessing, feature fusion, training, or evaluation code.

---

## 1) Script Inventory

## A. “Normal” scripts (root + `scripts/`)

| Script | Type | Primary Role |
|---|---|---|
| `preprocess_dataset.py` | Python | Chunked CSV preprocessing and normalization to `text`/`label` schema |
| `make_sample.py` | Python | Fast stratified sampling for quick experiments |
| `train_baseline.py` | Python | End-to-end text model training/dry-run tokenization with RoBERTa |
| `check_transformers.py` | Python | Dependency smoke-check for `transformers` import |
| `validate_cleaned.py` | Python | Data sanity check (columns, class distribution, sample rows) |
| `scripts/wait_for_training_and_extract.ps1` | PowerShell | Waits for training PID and extracts latest checkpoint trainer state |

## B. Scripts in `multimodal/`

| File | Type | Role |
|---|---|---|
| `multimodal/README.md` | Markdown | External reference link only |

**Observation:** No `.py`, `.ipynb`, shell, or PowerShell scripts currently exist under `multimodal/`.

---

## 2) Functional Comparison

| Capability | Main scripts | `multimodal/` scripts |
|---|---|---|
| Dataset preprocessing | ✅ Implemented (`preprocess_dataset.py`) | ❌ Not implemented |
| Dataset validation | ✅ Implemented (`validate_cleaned.py`) | ❌ Not implemented |
| Sampling for fast iteration | ✅ Implemented (`make_sample.py`) | ❌ Not implemented |
| Model training | ✅ Implemented (`train_baseline.py`) | ❌ Not implemented |
| Experiment variants | ✅ Parent-context and subreddit variants via flags | ❌ Not implemented |
| Dependency health check | ✅ `check_transformers.py` | ❌ Not implemented |
| Training process monitoring | ✅ PowerShell watcher script | ❌ Not implemented |
| Documentation of runnable multimodal workflow | ⚠️ Partial (overall README text-focused) | ❌ No runnable workflow docs |

---

## 3) Architectural Differences

## Main script architecture (currently active)

The active pipeline follows a clear sequence:
1. **Raw CSV cleanup** (`preprocess_dataset.py`)
2. **Optional small-sample creation** (`make_sample.py`)
3. **Data sanity checks** (`validate_cleaned.py`)
4. **Tokenizer dry run / training** (`train_baseline.py`)
5. **Checkpoint monitoring/reporting** (`wait_for_training_and_extract.ps1`)

This supports practical experimentation and reproducibility for text-only sarcasm detection.

## `multimodal/` architecture (currently inactive)

`multimodal/` contains no executable pipeline components. There is no code for:
- image/audio/video feature loading
- text + non-text feature fusion
- multimodal dataset alignment
- multimodal model training/evaluation
- multimodal inference/export

So the folder is documentation-only at this stage.

---

## 4) Input/Output and Data Contract Comparison

## Main scripts

- **Input assumption:** tabular CSV with label and text-like fields.
- **Normalization:** converts selected columns to canonical `text`, `label`.
- **Output artifacts:** cleaned CSVs, sampled CSVs, and HuggingFace training outputs/checkpoints.
- **Failure handling:** explicit column checks and coercion logic for labels.

## `multimodal/`

- No explicit input contracts are defined in executable code.
- No outputs, checkpoints, or metrics generation scripts are present.

---

## 5) Maturity Comparison

| Dimension | Main scripts | `multimodal/` |
|---|---|---|
| Implementation completeness | High | Very low (placeholder) |
| Reproducibility | Moderate to high (documented command flow) | Not reproducible yet |
| Operational readiness | Usable now for text baseline | Not operational |
| Extensibility | Has CLI flags and modular script steps | Requires initial scaffold |

---

## 6) Risk and Gap Analysis

## Current strengths of normal scripts
- End-to-end runnable baseline exists.
- Practical utilities for validation and checkpoint inspection.
- Supports lightweight experimentation via sample and dry-run modes.

## Key multimodal gaps
- No baseline multimodal trainer to compare against text-only baseline.
- No alignment scripts to merge text with additional modalities.
- No standardized multimodal data schema.
- No multimodal-specific evaluation and ablation tooling.

## Impact
Without scripts under `multimodal/`, any multimodal experimentation depends on external/off-repo assets and cannot be reproduced directly from this codebase.

---

## 7) What This Means for Contributors

- If your objective is **text-only sarcasm detection**, use the main scripts immediately.
- If your objective is **multimodal sarcasm detection**, the repository currently lacks runnable multimodal code and needs initial implementation before experiments can be reproduced.

---

## 8) Suggested Next Steps for `multimodal/` (Documentation-Level Roadmap)

To close the parity gap with the main scripts, `multimodal/` should eventually include:
1. a multimodal preprocessing script
2. a multimodal sample-builder
3. a multimodal training entrypoint
4. a multimodal validation/evaluation script
5. a reproducible quick-run command set similar to the root README

This would make side-by-side comparison (text-only vs multimodal) measurable and reproducible within the repository itself.
