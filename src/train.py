# train.py
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import yaml
from prepare_dataset import MNISTEvenOddDatasetManager
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    TrainingArguments,
    Trainer
)
import numpy as np
import evaluate
import torch
from datasets import load_from_disk, Dataset
from pathlib import Path  

def load_config(config_path="config/training.yaml"):
    """Load training config from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, f1."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="macro")["f1"],
        "precision": precision.compute(predictions=predictions, references=labels, average="macro")["precision"],
        "recall": recall.compute(predictions=predictions, references=labels, average="macro")["recall"],
    }


def collate_fn(batch):
    """Custom data collator for image classification."""
    # Convert pixel_values to tensors if they're lists/numpy arrays
    pixel_values = []
    for x in batch:
        pixel_data = x["pixel_values"]
        if isinstance(pixel_data, list):
            # Convert list to tensor
            pixel_data = torch.tensor(pixel_data)
        elif isinstance(pixel_data, np.ndarray):
            # Convert numpy array to tensor
            pixel_data = torch.from_numpy(pixel_data)
        pixel_values.append(pixel_data)
    
    pixel_values = torch.stack(pixel_values)
    labels = torch.tensor([x["label"] for x in batch])
    return {"pixel_values": pixel_values, "labels": labels}

def main():
    #Load configs
    train_config = load_config("config/training.yaml")
    dataset_config_path = "config/dataset.yaml"

    #Load Datasets
    train_ds = load_from_disk(train_config["training"]["train_dataset"])
    val_ds = load_from_disk(train_config["training"]["val_dataset"])
    test_ds = load_from_disk(train_config["training"]["test_dataset"])

    #REDUCE DATASET SIZE FOR FASTER TRAINING
    train_ds = train_ds.select(range(5000))    # 5K instead of 54K
    val_ds = val_ds.select(range(1000))        # 1K instead of 6K


    # Load model + processor
    model = AutoModelForImageClassification.from_pretrained(
        train_config["model"]["pretrained_name"],
        num_labels=2,
        id2label=train_config["model"]["id2label"],
        label2id=train_config["model"]["label2id"],
        ignore_mismatched_sizes=True
    )

    processor = AutoImageProcessor.from_pretrained(train_config["model"]["pretrained_name"])

    #Training arguments
    training_args = TrainingArguments(
        output_dir=train_config["training"]["output_dir"],
        eval_strategy=train_config["training"]["eval_strategy"],  
        save_strategy=train_config["training"]["save_strategy"],
        learning_rate=float(train_config["training"]["learning_rate"]),  
        per_device_train_batch_size=train_config["training"]["batch_size"],
        per_device_eval_batch_size=train_config["training"]["batch_size"],
        num_train_epochs=train_config["training"]["epochs"],
        weight_decay=train_config["training"]["weight_decay"],
        logging_dir=train_config["training"]["logging_dir"],
        eval_steps=train_config["training"]["eval_steps"], 
        save_steps=train_config["training"]["save_steps"], 
        logging_steps=100,  # More reasonable than 10
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,  
        push_to_hub=False,
        report_to=None,                 # Disable external logging
        no_cuda=True,                    #  Force CPU
        dataloader_pin_memory=False,     # Disable pin memory on Mac
        dataloader_num_workers=0,        # Single process data loading
    )

    #Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,   # processor = "tokenizer" for images
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Evaluate
    results = trainer.evaluate(test_ds)
    print("Test Results:", results)

    # Save final model
    trainer.save_model(train_config["training"]["save_dir"])
    processor.save_pretrained(train_config["training"]["save_dir"])
    print(f"Model saved at {train_config['training']['save_dir']}")


if __name__ == "__main__":
    main()