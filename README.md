# MNIST Even/Odd Binary Classifier

A deep learning project that fine-tunes a ResNet-18 model to classify MNIST handwritten digits as **even** (0, 2, 4, 6, 8) or **odd** (1, 3, 5, 7, 9). The model achieves **99.22% accuracy** on the test set, demonstrating excellent performance on this binary classification task.

## 🎯 Overview

This project transforms the traditional 10-class MNIST digit classification problem into a binary classification task. By leveraging transfer learning with a pre-trained ResNet-18 model from Hugging Face

### Key Features


- **ResNet-18 Architecture**: Implements ResNet-18 for binary classification tasks
- **Hugging Face Integration**: Fully compatible with Transformers library and Trainer API
- **Model Compatibility**: Models can be easily loaded using `AutoModelForImageClassification.from_pretrained()`
- **Trainer API**: Training implemented using Hugging Face's efficient Trainer API
- **YAML Configuration**: Modular configuration system for dataset and training parameters
- **Binary Classification**: Classifies digits as even (0, 2, 4, 6, 8) or odd (1, 3, 5, 7, 9)
- **Comprehensive Metrics**: Tracks multiple evaluation metrics, including accuracy, precision, recall, and F1 score


### Performance

- **Accuracy**: 99.22%
- **F1 Score**: 99.22%
- **Precision**: 99.22%
- **Recall**: 99.22%
- **Error Rate**: 0.78%

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Dataset Details](#dataset-details)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Inference & Evaluation](#inference--evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Steps

1. **Clone the repository**

```bash
git clone https://github.com/sohilkhan-2611/restnet_binary_classification.git
cd mnist-even-odd-classifier
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Required Dependencies

The project uses the following major libraries:

- `torch` - PyTorch deep learning framework
- `transformers` - Hugging Face Transformers (ResNet-18, AutoImageProcessor)
- `datasets` - Hugging Face Datasets library
- `evaluate` - Evaluation metrics (accuracy, F1, precision, recall)
- `matplotlib` - Visualization
- `numpy` - Numerical operations
- `PyYAML` - Configuration file parsing

## 📁 Project Structure

```
project/
├── config/
│   ├── training.yaml             # Training hyperparameters and settings
│   └── dataset.yaml              # Dataset paths and preprocessing settings
├── src/
│   ├── prepare_dataset.py        # Dataset preparation script
│   ├── train.py                  # Main training script
│   └── inference.py              # Inference and testing script
├── data/
│   └── processed/                # Processed dataset (generated)
│       ├── test/
│       ├── train/
│       └── val/
├── outputs/
│   └── resnet18_even_odd/        # Training outputs (generated)
│       ├── final_model/          # Final trained model
│       ├── checkpoints/          # Training checkpoints
│       └── logs/                 # TensorBoard logs
├── saved_model/                  # Final saved model (generated)
├── results/                      # Performance reports (generated)
│   └── model_performance.txt     # Detailed metrics report
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## ⚙️ Configuration

The project uses YAML configuration files for easy customization:

### `config/dataset.yaml`

Controls dataset loading and preprocessing:

```yaml
dataset:
  name: "mnist"
  validation_split_ratio: 0.1
  random_seed: 42

model:
  pretrained_name: "microsoft/resnet-18"
  id2label: {0: "even", 1: "odd"}
  label2id: {"even": 0, "odd": 1}

data_processing:
  batch_size: 100
```

### `config/training.yaml`

Controls training hyperparameters:

```yaml
training:
  output_dir: "./outputs/resnet18_even_odd"
  save_dir: "./saved_model"
  train_dataset: "data/processed/train"
  val_dataset: "data/processed/val"
  test_dataset: "data/processed/test"
  epochs: 3
  batch_size: 32
  learning_rate: 2e-5
  weight_decay: 0.01
  eval_strategy: "steps"
  save_strategy: "steps"
  eval_steps: 500
  save_steps: 500
```

## 💻 Usage

### Step 1: Prepare the Dataset

Run the dataset preparation script to download MNIST, transform labels, and preprocess images:

```bash
python src/prepare_dataset.py
```

This script will:
- Download the MNIST dataset from Hugging Face
- Transform digit labels to binary (even=0, odd=1)
- Split data into train (54K), validation (6K), and test (10K) sets
- Preprocess images (resize to 224×224, convert to RGB, normalize)
- Save preprocessed datasets to `data/processed/`

**Expected Output:**
```
Loading MNIST dataset...
Transforming labels to even/odd binary classification...
Creating dataset splits...
Preprocessing images for ResNet-18...
✅ Preprocessed datasets saved to disk at 'data/processed'
```

### Step 2: Train the Model

Train the ResNet-18 model on the preprocessed dataset:

```bash
python src/train.py
```

**Training Features:**
- Fine-tunes ResNet-18 on binary classification
- Saves checkpoints every 500 steps
- Evaluates on validation set every 500 steps
- Automatically saves the best model based on accuracy
- Generates TensorBoard logs for monitoring

**Note:** The provided training script uses a reduced dataset (5K training, 1K validation) for faster experimentation. Remove these lines for full dataset training:

```python
# In train.py, comment out or remove:
train_ds = train_ds.select(range(5000))
val_ds = val_ds.select(range(1000))
```

### Step 3: Evaluate the Model

Run inference on the test set and generate performance reports:

```bash
python src/inference.py
```

**Output:**
```
Test results: {'eval_loss': 0.027233, 'eval_accuracy': 0.9922, ...}

==================================================
PERFORMANCE SUMMARY:
==================================================
Accuracy:  99.22%
F1 Score:  99.22%
Precision: 99.22%
Recall:    99.22%
Loss:      0.027233
==================================================

Performance results saved to 'results/model_performance.txt'


## 📊 Dataset Details

### MNIST Dataset

- **Source**: Hugging Face Datasets (`mnist`)
- **Original Classes**: 10 digits (0-9)
- **Transformed Classes**: 2 labels (even/odd)
- **Total Samples**: 70,000
  - Training: 54,000 samples
  - Validation: 6,000 samples
  - Testing: 10,000 samples

### Label Transformation

| Original Digits | Binary Label | Class Name |
|----------------|--------------|------------|
| 0, 2, 4, 6, 8  | 0            | Even       |
| 1, 3, 5, 7, 9  | 1            | Odd        |

### Image Preprocessing

1. **Grayscale to RGB**: Convert single-channel MNIST images to 3-channel RGB
2. **Resize**: Scale from 28×28 to 224×224 (ResNet-18 input size)
3. **Normalization**: Apply ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
4. **Tensor Conversion**: Convert to PyTorch tensors

## 🏗️ Model Architecture

### ResNet-18

- **Base Model**: Microsoft's pre-trained ResNet-18 (`microsoft/resnet-18`)
- **Modification**: Final classification layer modified from 1000 classes to 2 classes
- **Parameters**: ~11 million (most frozen, only final layers fine-tuned)
- **Input Size**: 224×224×3 RGB images
- **Output**: Binary classification (even/odd)

### Transfer Learning Strategy

1. Load pre-trained ResNet-18 weights (trained on ImageNet)
2. Replace final fully-connected layer for binary classification
3. Fine-tune the entire model on MNIST even/odd task
4. Use low learning rate (2e-5) to preserve pre-trained features

## 🎓 Training Process

### Hyperparameters

- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Weight Decay**: 0.01
- **Batch Size**: 32
- **Epochs**: 3
- **Evaluation Frequency**: Every 500 steps

### Training Pipeline

1. Load preprocessed datasets from disk
2. Initialize ResNet-18 with modified classification head
3. Train with cross-entropy loss
4. Evaluate on validation set at regular intervals
5. Save checkpoints and best model
6. Generate training logs for TensorBoard

### Hardware Requirements

- **CPU Training**: Fully supported (no CUDA required)
- **GPU Training**: Optional, significantly faster with CUDA
- **Memory**: ~4GB RAM minimum
- **Disk Space**: ~500MB for datasets and models

## 🔬 Inference & Evaluation

### Evaluation Metrics

The model is evaluated using multiple metrics:

- **Accuracy**: Overall correctness (99.22%)
- **F1 Score**: Harmonic mean of precision and recall (99.22%)
- **Precision**: True positives / (True positives + False positives) (99.22%)
- **Recall**: True positives / (True positives + False negatives) (99.22%)
- **Loss**: Cross-entropy loss (0.027233)

### Performance Report

The inference script generates a detailed report at `results/model_performance.txt`

## 📈 Results

### Test Set Performance

- **Test Samples**: 10,000
- **Correct Predictions**: 9,922
- **Incorrect Predictions**: 78
- **Error Rate**: 0.78%

### Key Insights

1. **Balanced Performance**: Equal precision and recall indicate no class imbalance issues
2. **High Confidence**: Low loss (0.027) suggests confident predictions
3. **Generalization**: Excellent test accuracy shows the model generalizes well
4. **Efficiency**: Fast training time (~minutes on CPU for reduced dataset)

