import os
import evaluate
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score
from make_dataset import KaggleMovieReviewsDataset

# --- 1. Setup & VRAM Management ---
output_dir = "final_model"
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"  

# --- 2. Load Dataset ---
# We use the same dataset class because your context logic (adding full sentence) is excellent
print(f"Loading Data for {MODEL_NAME}...")
dataset_tool = KaggleMovieReviewsDataset(
    train_tsv_path='../resources/sentiment-analysis-on-movie-reviews/train.tsv',
    test_tsv_path='../resources/sentiment-analysis-on-movie-reviews/test.tsv',
    train_pct=0.85, 
    seed=42, 
    add_context_columns=True, # Keep this! Context helps DeBERTa too.
    context_dropout_p=0.2
)
ds_splits = dataset_tool.get_train_datasetdict()
test_dataset = dataset_tool.get_kaggle_test_dataset()

# Helper to swap context column
def use_context(ds):
    if "text_with_context" in ds.column_names:
        return ds.remove_columns([c for c in ["text"] if c in ds.column_names]).rename_column("text_with_context", "text")
    return ds

train_ds = use_context(ds_splits["train"]).shuffle(seed=42)
eval_ds  = use_context(ds_splits["test"])

# rename label column to labels for Trainer compatibility
train_ds = train_ds.rename_column("label", "labels")
eval_ds = eval_ds.rename_column("label", "labels")


# --- 3. Tokenization ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

print("Tokenizing...")
tokenized_train_ds = train_ds.map(preprocess_function, batched=True)
tokenized_eval_ds = eval_ds.map(preprocess_function, batched=True)
tokenized_test = use_context(test_dataset).map(preprocess_function, batched=True)

# --- 4. Model Setup ---

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=5
)



# --- 6. Training Arguments ---
training_args = TrainingArguments(
    output_dir="deberta_new",
    learning_rate=4e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    warmup_steps=100,
    fp16=False,
	bf16=False,
    weight_decay=0,
    max_grad_norm=1.0,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_eval_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# --- 7. Train ---
print("Starting Training...")
trainer.train()

# --- 8. Predict & Submit ---
print("Predicting on Test Set...")
preds = trainer.predict(tokenized_test)
final_preds = np.argmax(preds.predictions, axis=1)

submission = pd.DataFrame({
    "PhraseId": test_dataset["PhraseId"],
    "Sentiment": final_preds
})

submission.to_csv("submission_deberta.csv", index=False)
print("Saved submission_deberta.csv")