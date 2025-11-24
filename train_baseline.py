import argparse
import logging
import math
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# We'll import `transformers` inside `main()` to ensure the environment
# is ready and to provide a clearer error message if the package is missing.
AutoTokenizer = None
AutoModelForSequenceClassification = None
DataCollatorWithPadding = None
Trainer = None
TrainingArguments = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def load_and_prepare(input_csv: str, text_col: str = "text", label_col: str = "label", max_samples: Optional[int] = None, use_parent: bool = False, use_subreddit: bool = False):
    df = pd.read_csv(input_csv, engine="python")
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Expected columns '{text_col}' and '{label_col}' in {input_csv}. Found: {df.columns.tolist()}")

    if use_parent:
        if 'parent_comment' not in df.columns:
            raise ValueError(f"--use_parent requires 'parent_comment' column in {input_csv}. Found: {df.columns.tolist()}")
        df = df[[text_col, label_col, 'parent_comment']].copy()
        # Concatenate parent_comment and text with markers
        df['parent_comment'] = df['parent_comment'].fillna('')
        df[text_col] = df.apply(
            lambda row: f"[PARENT] {row['parent_comment']} [CHILD] {row[text_col]}" if row['parent_comment'] else row[text_col],
            axis=1
        )
        df = df[[text_col, label_col]].dropna()
    elif use_subreddit:
        if 'subreddit' not in df.columns:
            raise ValueError(f"--use_subreddit requires 'subreddit' column in {input_csv}. Found: {df.columns.tolist()}")
        df = df[[text_col, label_col, 'subreddit']].copy()
        # Prepend subreddit as special token
        df['subreddit'] = df['subreddit'].fillna('unknown')
        df[text_col] = df.apply(
            lambda row: f"[SR:{row['subreddit']}] {row[text_col]}",
            axis=1
        )
        df = df[[text_col, label_col]].dropna()
    else:
        df = df[[text_col, label_col]].dropna()
    
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    if max_samples is not None and max_samples > 0:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)

    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default="outputs/baseline")
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--dry_run", action="store_true", help="Only tokenize a small sample and exit (no model downloads)")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max rows to load for dry-run or quick tests")
    parser.add_argument("--use_parent", action="store_true", help="Concatenate parent_comment with text for context")
    parser.add_argument("--use_subreddit", action="store_true", help="Prepend subreddit as special token [SR:subreddit]")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Apply max_samples for both dry_run and quick tests (when max_samples is explicitly set and not the default 1000 for dry-run)
    use_max_samples = args.max_samples if (args.dry_run or args.max_samples != 1000) else None
    df = load_and_prepare(args.input_csv, max_samples=use_max_samples, use_parent=args.use_parent, use_subreddit=args.use_subreddit)
    logger.info("Loaded rows: %d", len(df))
    logger.info("Label distribution: %s", df['label'].value_counts().to_dict())

    if args.dry_run:
        try:
            from transformers import AutoTokenizer
        except Exception:
            logger.error("Transformers not installed; cannot dry-run tokenization. Run: pip install transformers datasets")
            return

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        examples = df['text'].tolist()[: min(100, len(df))]
        tokenized = tokenizer(examples, truncation=True, padding=True, max_length=args.max_length)
        token_lengths = [len(ids) for ids in tokenized['input_ids']]
        avg_length = sum(token_lengths) / len(token_lengths)
        max_length_found = max(token_lengths)
        logger.info("Tokenized %d examples. Avg length: %.1f, Max length: %d", len(examples), avg_length, max_length_found)
        logger.info("Example input_ids length: %d", len(tokenized['input_ids'][0]))
        # show a tokenized example
        logger.info("Sample tokens: %s", tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0])[:30])
        if args.use_parent:
            logger.info("Sample concatenated text (first 200 chars): %s", examples[0][:200])
        if args.use_subreddit:
            logger.info("Sample subreddit-prepended text (first 200 chars): %s", examples[0][:200])
        return

    # Full training path (user can run this if they have transformers and a GPU/CPU ready)
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except Exception:
        raise RuntimeError("Transformers library required for training. Install with: pip install transformers datasets")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # use datasets in-memory; perform a stratified split using sklearn then tokenize
    from datasets import Dataset

    train_df, test_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    def preprocess(batch):
        return tokenizer(batch['text'], truncation=True, padding=False, max_length=args.max_length)

    tokenized_train = train_ds.map(preprocess, batched=True)
    tokenized_test = test_ds.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
