import torch
from datasets import load_from_disk
from transformers import AutoModelForImageClassification, AutoImageProcessor, Trainer
import numpy as np
import evaluate
import os
from datetime import datetime

# Load model & processor
save_dir = "./saved_model"
model = AutoModelForImageClassification.from_pretrained(save_dir, local_files_only=True)
processor = AutoImageProcessor.from_pretrained(save_dir, local_files_only=True)

# Moves the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Load preprocessed dataset
test_dataset = load_from_disk("data/processed/test")

def collate_fn(batch):
    pixel_values = []
    labels = []
    
    for x in batch:
        pixel_data = x["pixel_values"]
        
        # Convert list or ndarray to tensor if needed
        if isinstance(pixel_data, list):
            pixel_data = torch.tensor(pixel_data, dtype=torch.float32)
        elif isinstance(pixel_data, np.ndarray):
            pixel_data = torch.from_numpy(pixel_data).float()
        elif isinstance(pixel_data, torch.Tensor):
            pixel_data = pixel_data.float()
        else:
            raise TypeError(f"Unsupported type for pixel_values: {type(pixel_data)}")
        
        pixel_values.append(pixel_data)
        labels.append(x["label"])
    
    # Stack into batch tensor
    pixel_values = torch.stack(pixel_values)
    labels = torch.tensor(labels, dtype=torch.long)  # Use long for classification labels
    
    return {"pixel_values": pixel_values, "labels": labels}


from torch.utils.data import DataLoader

loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)

batch = next(iter(loader))
print("Batch pixel_values shape:", batch["pixel_values"].shape)
print("Batch labels shape:", batch["labels"].shape)
print("Pixel values type:", type(batch["pixel_values"]))
print("Labels type:", type(batch["labels"]))