from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set, Dict

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer


@dataclass
class KaggleMovieReviewsDataset:
    """
    TSV -> HF Dataset
    Rename columns -> {text, label}
    SentenceId-group split -> DatasetDict({"train","test"})
    Add:
      - full_sentence (longest phrase per SentenceId by default)
      - text_with_context = phrase + sep_token + full_sentence
      - optional context dropout augmentation
    """
    train_tsv_path: str
    test_tsv_path: Optional[str] = None

    # Split config
    train_pct: float = 0.9
    seed: int = 42

    # Column config (Kaggle file schema)
    sentence_id_col: str = "SentenceId"
    phrase_id_col: str = "PhraseId"
    phrase_col: str = "Phrase"
    sentiment_col: str = "Sentiment"

    # Output column names
    text_col: str = "text"
    label_col: str = "label"

    # Context features
    add_context_columns: bool = True
    full_sentence_col: str = "full_sentence"
    text_with_context_col: str = "text_with_context"

    # How to choose "full sentence" per SentenceId
    # "longest" is a good heuristic for this dataset (full sentence tends to be the longest phrase).
    full_sentence_strategy: str = "longest"  # {"longest"}

    # Augmentation: sometimes drop context so model also learns to work on phrase-only inputs
    context_dropout_p: float = 0.0  # set e.g. 0.3 to enable

    # Separator token source: pass any HF tokenizer id used by your embedding model
    # Example: "sentence-transformers/all-mpnet-base-v2" or "roberta-base"
    sep_token_model_id: str = "roberta-base"

    def _load_tsv(self, path: str) -> Dataset:
        # TSV is a CSV variant with delimiter="\t"
        return load_dataset("csv", data_files={"data": path}, delimiter="\t")["data"]

    def _rename_columns(self, ds: Dataset) -> Dataset:
        # Phrase -> text
        if self.phrase_col in ds.column_names and self.text_col not in ds.column_names:
            ds = ds.rename_column(self.phrase_col, self.text_col)

        # Sentiment -> label (only exists for train)
        if self.sentiment_col in ds.column_names and self.label_col not in ds.column_names:
            ds = ds.rename_column(self.sentiment_col, self.label_col)

        return ds

    def _split_by_sentence_id(self, ds: Dataset) -> DatasetDict:
        if not (0.0 < self.train_pct < 1.0):
            raise ValueError(f"train_pct must be in (0,1). Got {self.train_pct}")

        if self.sentence_id_col not in ds.column_names:
            raise ValueError(f"Missing '{self.sentence_id_col}' in dataset columns: {ds.column_names}")

        sids = np.array(ds.unique(self.sentence_id_col))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(sids)

        cut = int(len(sids) * self.train_pct)
        train_sids: Set[int] = set(map(int, sids[:cut]))
        test_sids: Set[int] = set(map(int, sids[cut:]))

        train_ds = ds.filter(lambda x: int(x[self.sentence_id_col]) in train_sids)
        test_ds  = ds.filter(lambda x: int(x[self.sentence_id_col]) in test_sids)

        return DatasetDict({"train": train_ds, "test": test_ds})

    def _build_full_sentence_map(self, ds: Dataset) -> Dict[int, str]:
        """
        Returns: {SentenceId: full_sentence_text}
        Strategy: 'longest' phrase by character length.
        """
        if self.full_sentence_strategy != "longest":
            raise ValueError(f"Unsupported full_sentence_strategy: {self.full_sentence_strategy}")

        # Pull only needed columns to python lists (fast enough for this dataset size)
        sids = ds[self.sentence_id_col]
        phrases = ds[self.text_col]

        best: Dict[int, str] = {}
        best_len: Dict[int, int] = {}

        for sid, phr in zip(sids, phrases):
            sid_int = int(sid)
            phr_str = "" if phr is None else str(phr)
            L = len(phr_str)
            if (sid_int not in best_len) or (L > best_len[sid_int]):
                best_len[sid_int] = L
                best[sid_int] = phr_str

        return best

    def _add_context_columns(self, ds: Dataset) -> Dataset:
        """
        Adds:
          - full_sentence
          - text_with_context
        Optionally applies context dropout augmentation.
        """
        # sep_token for RoBERTa-family is tokenizer.sep_token (often "</s>") :contentReference[oaicite:4]{index=4}
        tok = AutoTokenizer.from_pretrained(self.sep_token_model_id, use_fast=True)
        sep = tok.sep_token or "</s>"

        full_map = self._build_full_sentence_map(ds)
        rng = np.random.default_rng(self.seed)

        def add_cols(batch):
            sids = batch[self.sentence_id_col]
            phrases = batch[self.text_col]

            fulls = []
            combined = []

            for sid, phr in zip(sids, phrases):
                sid_int = int(sid)
                phr_str = "" if phr is None else str(phr)
                full = full_map.get(sid_int, "")
                fulls.append(full)

                use_context = True
                if self.context_dropout_p > 0.0:
                    # with prob p, drop context (phrase-only)
                    if rng.random() < self.context_dropout_p:
                        use_context = False

                if use_context and full:
                    combined.append(f"{phr_str} {sep} {full}")
                else:
                    combined.append(phr_str)

            batch[self.full_sentence_col] = fulls
            batch[self.text_with_context_col] = combined
            return batch

        return ds.map(add_cols, batched=True)

    def get_train_datasetdict(self) -> DatasetDict:
        """
        Loads train TSV, renames columns, adds context columns (optional),
        and splits by SentenceId into DatasetDict({"train","test"}).
        """
        ds = self._load_tsv(self.train_tsv_path)
        ds = self._rename_columns(ds)

        if self.add_context_columns:
            ds = self._add_context_columns(ds)

        return self._split_by_sentence_id(ds)

    def get_kaggle_test_dataset(self) -> Optional[Dataset]:
        """
        Loads Kaggle test TSV (no labels), renames Phrase->text,
        adds context columns (optional) so inference can match training format.
        """
        if not self.test_tsv_path:
            return None

        ds = self._load_tsv(self.test_tsv_path)
        ds = self._rename_columns(ds)

        if self.add_context_columns:
            ds = self._add_context_columns(ds)

        return ds
