# Sarcasm Detection — Reproducible setup and quick baseline

This repository contains preprocessing and a small baseline training pipeline for sarcasm detection using RoBERTa.

This README shows exact commands (PowerShell) to create a venv, install dependencies (including CPU PyTorch wheel), and reproduce the quick experiments I ran locally.

Prerequisites
- Python 3.9+ installed and on PATH
- Powershell (Windows) or a POSIX shell (adjust activation commands accordingly)

1) Create and activate virtual environment (PowerShell)

If you are using PowerShell and your system prevents running Activate.ps1 due to execution policy, either run PowerShell as Administrator and change the execution policy or use the process-scoped relaxed policy shown below. The commands below use the process-scoped policy so they do not change system-wide settings.

```powershell
# from repository root (e.g. d:\Sarcasm Detection)
python -m venv .venv
# allow script execution for this session only (optional if Activate.ps1 is blocked)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
# activate venv (PowerShell)
.\.venv\Scripts\Activate.ps1
# Alternative (cmd.exe): 
# .\.venv\Scripts\activate.bat
```

2) Upgrade pip and install CPU PyTorch wheel (recommended for reproducibility on CPU)

The `torch` package is platform-specific. To install a CPU-only PyTorch wheel reproducibly on Windows, run the command below before installing the rest of `requirements.txt`.

```powershell
# upgrade pip first
python -m pip install -U pip
# install CPU-only torch (this uses the official PyTorch CPU wheel index)
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
```

If you have a GPU and want CUDA-enabled PyTorch, refer to https://pytorch.org for the exact `pip` command matching your CUDA version (do NOT install the CPU wheel in that case).

3) Install the rest of Python dependencies

```powershell
pip install -r requirements.txt
```

4) Reproduce the quick sample and training runs

- Create a smaller sample (~2k rows) for fast experiments:

```powershell
python make_sample.py
# creates: d:\Sarcasm Detection\train_sample.csv
```

- Dry-run tokenization (no model download; useful to verify pipeline):

```powershell
python train_baseline.py --input_csv "d:\Sarcasm Detection\train-balanced-sarcasm.cleaned.csv" --dry_run --max_samples 200
```

- Quick 1-epoch baseline training on the sampled dataset (this matches the quick experiment):

```powershell
python train_baseline.py --input_csv "d:\Sarcasm Detection\train_sample.csv" --output_dir "d:\Sarcasm Detection\outputs\baseline_quick" --model_name roberta-base --epochs 1 --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --max_length 128
```

- Full 3-epoch baseline on the full cleaned dataset (longer run):

```powershell
python train_baseline.py --input_csv "d:\Sarcasm Detection\train-balanced-sarcasm.cleaned.csv" --output_dir "d:\Sarcasm Detection\outputs\baseline_full" --model_name roberta-base --epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --max_length 256
```

Notes and tips
- If `Activate.ps1` is blocked by execution policy, the `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force` command above will allow activation for the current session.
- If you prefer not to change execution policy, open a `cmd.exe` terminal and run `\.venv\Scripts\activate.bat` instead.
- If you want to use a smaller model for faster iteration replace `--model_name roberta-base` with a tiny model such as `sshleifer/tiny-roberta`.
- To reproduce the exact CPU environment I used earlier, install the `httpx` upgrade if you encounter dataset import errors: `python -m pip install -U httpx`.

Reproducible outputs
- Trained model checkpoints and trainer state for the quick run are saved under `d:\Sarcasm Detection\outputs\baseline_quick`.
- The sample created by `make_sample.py` is `d:\Sarcasm Detection\train_sample.csv`.

If you want, I can:
- Run the 3-epoch baseline for you here (will take longer) and return final metrics and best checkpoint path.
- Add scripts to run the parent-comment experiment and compare metrics automatically.