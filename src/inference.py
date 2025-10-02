# test_model.py
import torch
from datasets import load_from_disk
from transformers import AutoModelForImageClassification, AutoImageProcessor, Trainer
import numpy as np
import evaluate

# Load model & processor
save_dir = "saved_model"
model = AutoModelForImageClassification.from_pretrained(save_dir)
processor = AutoImageProcessor.from_pretrained(save_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Load preprocessed dataset
test_dataset = load_from_disk("data/processed/test")

# Simple collate function
def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch])
    return {"pixel_values": pixel_values, "labels": labels}

# Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        "precision": precision.compute(predictions=preds, references=labels, average="macro")["precision"],
        "recall": recall.compute(predictions=preds, references=labels, average="macro")["recall"],
    }

# Evaluate
trainer = Trainer(model=model, tokenizer=processor, data_collator=collate_fn, compute_metrics=compute_metrics)
results = trainer.evaluate(test_dataset)
print("Test results:", results)

# Single sample prediction
sample = test_dataset[0]
pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
with torch.no_grad():
    pred = torch.argmax(model(pixel_values=pixel_values).logits, dim=-1).item()
print(f"True label: {sample['label']}, Predicted: {pred}")