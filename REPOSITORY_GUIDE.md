# Repository File Guide

This document explains the files and directories in this repository and how
they fit into the sarcasm-detection workflow.

## Project overview

The project fine-tunes transformer language models to classify Reddit comments
as sarcastic or non-sarcastic. There are two high-level training approaches:

1. **Text-only language-model fine-tuning**: implemented locally in
   `train_baseline.py` with Hugging Face Transformers. The model is based on
   `roberta-base` by default and predicts one of two labels from comment text.
   The same training script can also run quick experiments that add parent
   comment context or subreddit metadata to the text.
2. **Multimodal training**: represented by `multimodal/README.md` and an
   external Google Drive link. This checkout does not contain an executable
   multimodal model, dataset loader, or training script, so the multimodal
   approach cannot be reproduced from the local source files alone.

The documented experiments found that the text-only baseline was the strongest
local approach. See `RESULTS.md` for the recorded metrics and comparison of
the baseline, parent-context, and subreddit-conditioned variants.

## Typical data and training flow

The intended pipeline is:

```text
train-balanced-sarcasm.csv
        |
        v
preprocess_dataset.py
        |
        v
train-balanced-sarcasm.cleaned.csv
        |
        +--> make_sample.py --> train_sample.csv   (quick experiments)
        |
        v
train_baseline.py
        |
        +--> outputs/baseline_full/
        +--> outputs/baseline_quick/
        +--> outputs/baseline_parent_quick*/
        +--> outputs/baseline_subreddit_quick/
```

The training script loads a CSV containing `text` and `label`, performs a
stratified 90/10 train/test split, tokenizes the text, fine-tunes a
two-class sequence-classification model, evaluates at epoch boundaries, saves
checkpoints, and stores the best model according to F1 score.

## Root-level files

### `README.md`

The primary setup and reproduction instructions. It documents:

- Python virtual-environment creation and activation.
- CPU-only PyTorch installation for the original Windows/PowerShell workflow.
- Installation from `requirements.txt`.
- Creation of the 2,000-row sample dataset.
- Dry-run tokenization.
- Quick one-epoch training.
- Full three-epoch baseline training.
- The expected output directories and useful troubleshooting notes.

The commands use Windows paths and PowerShell syntax, but the Python scripts
themselves are generally runnable from other platforms with equivalent paths.

### `RESULTS.md`

The experiment report. It compares:

- **Baseline**: the comment text only.
- **Parent-context**: `[PARENT] parent_comment [CHILD] text`.
- **Subreddit-conditioning**: `[SR:subreddit] text`.

It records the baseline full-run result as the best recorded model
(F1 `0.7749`, accuracy `77.62%`) and records lower-scoring quick experiments
for the additional context features. It also lists hyperparameters,
recommendations, output directories, and reproduction commands.

### `train_baseline.py`

The main local training and evaluation program.

Important responsibilities:

- Defines `compute_metrics`, which calculates accuracy, precision, recall, and
  binary F1 using scikit-learn.
- Defines `load_and_prepare`, which reads a CSV, validates required columns,
  converts labels to integers, optionally samples rows, and constructs the
  requested input representation.
- Supports the text-only path by keeping only the text and label columns.
- Supports `--use_parent` by requiring `parent_comment` and concatenating it
  with the comment using `[PARENT]` and `[CHILD]` markers.
- Supports `--use_subreddit` by requiring `subreddit` and prepending an
  `[SR:...]` marker.
- Supports `--dry_run` for tokenizer checks without model training.
- Uses `AutoTokenizer`, `AutoModelForSequenceClassification`, `Dataset`,
  `Trainer`, and dynamic padding from Hugging Face.
- Uses a stratified split with a fixed random seed (`42`).
- Saves model files and checkpoints under `--output_dir`.

Useful command-line options include `--input_csv`, `--output_dir`,
`--model_name`, `--epochs`, batch-size options, `--max_length`,
`--max_samples`, `--use_parent`, `--use_subreddit`, and `--dry_run`.

Although the file is named `train_baseline.py`, it is also the implementation
for the parent-context and subreddit quick experiments. The additional modes
change the input text rather than introducing separate model architectures.

### `preprocess_dataset.py`

The large-file CSV cleaning step. It reads the input in pandas chunks
(100,000 rows by default), which avoids requiring the entire raw dataset in
memory. For each chunk it:

- Checks for the configured text and label columns.
- Keeps the required columns and any optional columns passed with
  `--keep-cols`.
- Drops rows missing text or labels.
- Converts labels to numeric integers and drops invalid labels.
- Renames the selected text and label columns to canonical `text` and `label`.
- Writes the first chunk with a header and appends later chunks without
  repeating the header.

The default input text column is `comment`; the default output schema is
therefore suitable for `train_baseline.py`.

### `make_sample.py`

Creates the small `train_sample.csv` used for quick experiments. It reads the
cleaned dataset, attempts to retain class proportions while selecting up to
2,000 rows with a fixed random seed, and writes the result. The input and
output paths are currently hard-coded to the original Windows project
directory, so they may need to be edited when running on another machine.

### `validate_cleaned.py`

A lightweight inspection utility for the cleaned CSV. It prints the column
names, label distribution, and first five rows. It accepts an optional file
path argument; otherwise it uses the original hard-coded Windows path.
It is useful for checking that preprocessing produced the expected schema and
class balance before training.

### `check_transformers.py`

A dependency smoke test. It imports `transformers`, prints the installed
version when successful, and prints a traceback when the import fails.

### `check.ipynb`

A one-cell Jupyter notebook that checks PyTorch CUDA availability, GPU count,
and the detected GPU name. It is an environment check, not a training
pipeline.

### `requirements.txt`

Pins the main Python dependencies used by the text-only training pipeline:

- `transformers`, `datasets`, `tokenizers`, and `huggingface-hub` for model
  loading, tokenization, and dataset handling.
- `torch` is intentionally not pinned here because the README installs a
  platform-specific CPU or CUDA build separately.
- `scikit-learn` and `evaluate` for splitting and metrics.
- `accelerate` for Hugging Face training support.
- `pandas` and `pyarrow` for data processing.
- `tqdm` for progress reporting.
- `httpx` for compatible HTTP behavior in the recorded environment.

### `scripts/wait_for_training_and_extract.ps1`

A PowerShell helper for monitoring a training process. It waits for a
specified process ID, finds the numerically latest `checkpoint-*` directory
under the selected output directory, and prints its `trainer_state.json`.
If no checkpoint is found, it lists the output directory files instead.

The default process ID and output path are examples from the original local
run and should be overridden for a new training job.

### `.gitattributes`

Configures Git LFS for large model and dataset artifacts:

- `*.safetensors`
- `*.pt`
- `*.pth`
- `*.bin`
- `*.csv`

These files are treated as binary LFS-managed files rather than ordinary
text. A full checkout requires Git LFS to be installed and fetched.

### `.gitignore`

Excludes Python bytecode, virtual environments, build/package directories,
Jupyter checkpoints, logs, IDE files, editor swap files, and common operating
system metadata from Git.

### `test_add.txt`

A small scratch/test file containing the text `test`. It is not used by the
training or data-processing pipeline.

## Dataset files

The repository tracks these datasets through Git LFS:

### `train-balanced-sarcasm.csv`

The raw balanced sarcasm dataset. It is the source input for preprocessing.
The working tree currently contains an LFS pointer for the large file, not the
expanded CSV contents.

### `train-balanced-sarcasm.cleaned.csv`

The cleaned and normalized dataset expected by the training scripts. Its
canonical columns are `text` and `label`; optional metadata such as
`parent_comment` and `subreddit` can also be retained for the corresponding
quick experiments. This file is also LFS-managed.

### `train_sample.csv`

A smaller LFS-managed sample generated by `make_sample.py`. It is intended
for fast dry runs and one-epoch experiments rather than final model training.

## `multimodal/`

### `multimodal/README.md`

Contains a Google Drive link to multimodal project material. It establishes
the multimodal training direction, but the repository does not currently
contain the implementation needed to run that approach locally.

## `outputs/`

`outputs/` contains trained-model artifacts and intermediate checkpoints.
These are not source code; they are the saved state produced by
`train_baseline.py`.

Each top-level experiment directory generally contains:

- `config.json`: model architecture and label-configuration metadata.
- `model.safetensors`: learned model weights.
- `training_args.bin`: serialized Hugging Face training arguments.
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
  `vocab.json`, and `merges.txt`: tokenizer files needed to reload the model.
- `checkpoint-<step>/`: periodic checkpoints, when checkpoint saving was
  enabled.

Checkpoint directories additionally contain:

- `optimizer.pt`: optimizer state for resuming training.
- `scheduler.pt`: learning-rate scheduler state.
- `rng_state.pth`: random-number-generator state.
- `trainer_state.json`: step, metric, and training-progress state.
- A copy of the model and tokenizer files for that checkpoint.

### `outputs/baseline_full/`

The full text-only baseline run. It contains the final model, three numbered
training checkpoints, and `final_metrics_extracted.txt` with extracted final
metrics. This is the best recorded model according to `RESULTS.md`.

### `outputs/baseline_quick/`

A quick text-only baseline run using the small sample dataset. It is useful
for validating the training pipeline without waiting for the full run.

### `outputs/baseline_parent_quick/`

A quick parent-context experiment with a maximum sequence length of 256.

### `outputs/baseline_parent_quick_384/`

A quick parent-context experiment with a maximum sequence length of 384,
allowing more room for the concatenated parent and child text.

### `outputs/baseline_subreddit_quick/`

A quick experiment that prepends subreddit metadata to each comment.

The `checkpoint-225` directories in the quick runs are saved training
checkpoints. The numeric suffix is the trainer step at which the checkpoint
was created; it is not a model version number.

## `.history/`

### `.history/.gitignore_20251124225039`
### `.history/.gitignore_20251124225751`

Timestamped historical copies of `.gitignore`. They are editor/history
artifacts and are not referenced by the application or training pipeline.

## Reproducing the local text-only approach

After installing Git LFS and fetching the datasets and model artifacts, the
usual sequence is:

```bash
python -m pip install -r requirements.txt
python preprocess_dataset.py \
  --input train-balanced-sarcasm.csv \
  --output train-balanced-sarcasm.cleaned.csv \
  --text-col comment \
  --label-col label \
  --keep-cols parent_comment,subreddit
python make_sample.py
python train_baseline.py \
  --input_csv train-balanced-sarcasm.cleaned.csv \
  --output_dir outputs/baseline_full \
  --model_name roberta-base \
  --epochs 3 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --max_length 256
```

For a quick parent-context or subreddit experiment, add `--use_parent` or
`--use_subreddit` and use `--max_samples 2000`. For a tokenizer-only check,
add `--dry_run`.

