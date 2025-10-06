# inference.py

"""
    This script handles loading a trained ResNet-18 model, evaluating it on the 
    preprocessed test dataset, computing key metrics, and saving a performance report.

"""
import torch
from datasets import load_from_disk
from transformers import AutoModelForImageClassification, AutoImageProcessor, Trainer
import numpy as np
import evaluate                 #for metrics like accuracy, F1, precision, recall.
import os
from datetime import datetime  #timestamp for saving performance reports.

# Load model & processor
save_dir = "./saved_model"
model = AutoModelForImageClassification.from_pretrained(save_dir,local_files_only=True)
processor = AutoImageProcessor.from_pretrained(save_dir,local_files_only=True)

#Moves the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()        #Sets model to evaluation mode to disable gradients.

# Load preprocessed dataset
test_dataset = load_from_disk("data/processed/test")    #Loads preprocessed test dataset (from prepare_dataset.py)

# collate function
def collate_fn(batch):
    pixel_values = []
    labels = []

    """
	Converts pixel values from list/ndarray to torch.Tensor
	Stacks batch tensors
	Collects labels as tensor
	Returns dictionary suitable for Hugging Face Trainer
    """

    for x in batch:
        pixel_data = x["pixel_values"]
        
        # Convert list or ndarray to tensor if needed
        if isinstance(pixel_data, list):
            pixel_data = torch.tensor(pixel_data, dtype=torch.float32)
        elif isinstance(pixel_data, np.ndarray):
            pixel_data = torch.from_numpy(pixel_data).float()
        # Ensure it's float tensor
        elif isinstance(pixel_data, torch.Tensor):
            pixel_data = pixel_data.float()
        
        pixel_values.append(pixel_data)
        labels.append(x["label"])
    
    # Stack into batch tensor
    pixel_values = torch.stack(pixel_values)
    labels = torch.tensor(labels)
    
    return {"pixel_values": pixel_values, "labels": labels}


    
# Metrics

""" Prepares metric calculators for evaluation """

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")


#Function to compute metrics
def compute_metrics(eval_pred):
    """ Computes accuracy, F1, precision, recall using predictions from the model """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        "precision": precision.compute(predictions=preds, references=labels, average="macro")["precision"],
        "recall": recall.compute(predictions=preds, references=labels, average="macro")["recall"],
    }

# Evaluate - Trainer automates evaluation
trainer = Trainer(model=model, tokenizer=processor, data_collator=collate_fn, compute_metrics=compute_metrics)
results = trainer.evaluate(test_dataset)
# Uses your collate_fn to batch data
# Computes metrics during evaluation.
# results will contain loss + metrics (accuracy, f1, precision, recall)
print("Test results:", results)



# Saves accuracy, F1, precision, recall, and loss into a report file.
def save_performance_results(results, filename="model_performance.txt"):

    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Full path for the file
    filepath = os.path.join(results_dir, filename)

    with open(filename, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("FINAL MODEL PERFORMANCE REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {save_dir}\n")
        f.write(f"Test Samples: {len(test_dataset)}\n")
        f.write("\n" + "=" * 30 + "\n")
        f.write("PERFORMANCE METRICS:\n")
        f.write("=" * 30 + "\n")
        f.write(f"Accuracy:  {results.get('eval_accuracy', 0):.4f} ({results.get('eval_accuracy', 0)*100:.2f}%)\n")
        f.write(f"F1 Score:  {results.get('eval_f1', 0):.4f} ({results.get('eval_f1', 0)*100:.2f}%)\n")
        f.write(f"Precision: {results.get('eval_precision', 0):.4f} ({results.get('eval_precision', 0)*100:.2f}%)\n")
        f.write(f"Recall:    {results.get('eval_recall', 0):.4f} ({results.get('eval_recall', 0)*100:.2f}%)\n")
        f.write(f"Loss:      {results.get('eval_loss', 0):.6f}\n")
        f.write("\n" + "=" * 30 + "\n")
        f.write("PERFORMANCE SUMMARY:\n")
        f.write("=" * 30 + "\n")
        
        accuracy_percent = results.get('eval_accuracy', 0) * 100
        if accuracy_percent >= 99:
            f.write("OUTSTANDING: Model performance is excellent\n")
        elif accuracy_percent >= 95:
            f.write("VERY GOOD: Model performs very well\n")
        elif accuracy_percent >= 90:
            f.write("GOOD: Model performance is acceptable\n")
        else:
            f.write("NEEDS IMPROVEMENT: Model requires optimization\n")
            
        f.write(f"Error Rate: {(1 - results.get('eval_accuracy', 0))*100:.2f}%\n")

# Save the results
save_performance_results(results)
print("Performance results saved to 'results/model_performance.txt'")

# Console summary
print("\n" + "="*50)
print("PERFORMANCE SUMMARY:")
print("="*50)
print(f"Accuracy:  {results.get('eval_accuracy', 0)*100:.2f}%")
print(f"F1 Score:  {results.get('eval_f1', 0)*100:.2f}%")
print(f"Precision: {results.get('eval_precision', 0)*100:.2f}%")
print(f"Recall:    {results.get('eval_recall', 0)*100:.2f}%")
print(f"Loss:      {results.get('eval_loss', 0):.6f}")
print("="*50)