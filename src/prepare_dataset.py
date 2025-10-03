# prepare_dataset.py
from datasets import load_dataset
from transformers import AutoImageProcessor
import matplotlib.pyplot as plt
import numpy as np
import yaml
from typing import Dict, Tuple, Any
from pathlib import Path
from datasets import Dataset

class MNISTEvenOddDatasetManager:
    """
    A class to manage MNIST dataset loading, transformation, and verification
    for binary classification of even vs odd digits.
    """
    
    def __init__(self, config_path: str = "/config/dataset.yaml"):
        """
        Initialize the dataset manager with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.dataset = None
        self.processor = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_dataset(self) -> None:
        """
        Load MNIST dataset from Hugging Face hub.
        
        Steps:
        1. Load original MNIST dataset
        2. Transform labels to binary (even=0, odd=1)
        3. Split into train/validation/test sets
        """
        print("Loading MNIST dataset...")
        
        # Load original MNIST dataset
        self.dataset = load_dataset(
            self.config['dataset']['name'],
            split=None  # Load all splits
        )
        
        print("Dataset loaded successfully!")
        self._print_dataset_info("Original dataset structure")
    
    def transform_labels(self) -> None:
        """
        Transform digit labels to binary classification (even vs odd).
        
        Transformation:
        - Even digits (0,2,4,6,8) → label 0
        - Odd digits (1,3,5,7,9) → label 1
        """
        print("Transforming labels to even/odd binary classification...")
        
        def label_transform(example):
            """Transform single example label."""
            example["label"] = example["label"] % 2  # even=0, odd=1
            return example
        
        # Apply transformation to all splits
        self.dataset = self.dataset.map(label_transform)
        
        print("Labels transformed successfully!")
        self._verify_label_distribution()
    
    def create_splits(self) -> None:
        """
        Create train/validation/test splits.
        
        Split strategy:
        - Train: 80% of original training data (54,000 samples)
        - Validation: 10% of original training data (6,000 samples)  
        - Test: Original test data (10,000 samples)
        """
        print("Creating dataset splits...")
        
        # Split original train into train/validation
        split_dataset = self.dataset["train"].train_test_split(
            test_size=self.config['dataset']['validation_split_ratio'],
            seed=self.config['dataset']['random_seed']
        )
        
        self.train_dataset = split_dataset["train"]
        self.val_dataset = split_dataset["test"]
        self.test_dataset = self.dataset["test"]
        
        print("Dataset splits created successfully!")
        self._print_split_info()
    
    def setup_processor(self) -> None:
        """Initialize the image processor for ResNet-18."""
        print("🔧 Setting up image processor...")
        
        self.processor = AutoImageProcessor.from_pretrained(
            self.config['model']['pretrained_name']
        )
        
        print("Image processor initialized!")
    
    def preprocess_images(self) -> None:
        """
        Preprocess images for ResNet-18 model.
        
        Processing includes:
        - Resizing to 224x224
        - Normalization
        - Channel conversion (grayscale to RGB)
        """
        print("Preprocessing images for ResNet-18...")
        
        def preprocess_function(examples):
            """Preprocess batch of images."""
            # Convert grayscale PIL images to RGB
            images = []
            for img in examples["image"]:
                # Convert grayscale to RGB by duplicating the channel
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
            
            # Process each image individually to avoid batching issues
            pixel_values_list = []
            for img in images:
                inputs = self.processor(
                    img, 
                    return_tensors="pt"
                )
                # Extract the single tensor and remove batch dimension
                pixel_values_list.append(inputs["pixel_values"].squeeze(0))
            
            examples["pixel_values"] = pixel_values_list
            return examples
        
        # Apply preprocessing to all splits
        self.train_dataset = self.train_dataset.map(
            preprocess_function, 
            batched=True,
            batch_size=self.config['data_processing']['batch_size']
        )
        self.val_dataset = self.val_dataset.map(
            preprocess_function, 
            batched=True,
            batch_size=self.config['data_processing']['batch_size']
        )
        self.test_dataset = self.test_dataset.map(
            preprocess_function, 
            batched=True,
            batch_size=self.config['data_processing']['batch_size']
        )
        
        print("Image preprocessing completed!")
    
    def verify_dataset(self) -> None:
        """Comprehensive verification of the dataset."""
        print("\n" + "="*50)
        print("🔍 DATASET VERIFICATION REPORT")
        print("="*50)
        
        self._print_dataset_structure()
        self._verify_label_distribution()
        self._check_sample_quality()
        
        print("All verifications passed!")
    
    def _print_dataset_info(self, title: str) -> None:
        """Print dataset structure information."""
        print(f"\n {title}:")
        print(f"   Type: {type(self.dataset)}")
        if hasattr(self.dataset, 'keys'):
            for split_name, split_data in self.dataset.items():
                print(f"   {split_name}: {len(split_data)} samples")
    
    def _print_split_info(self) -> None:
        """Print information about dataset splits."""
        print(f"\n Dataset Split Sizes:")
        print(f"   Train: {len(self.train_dataset):,} samples")
        print(f"   Validation: {len(self.val_dataset):,} samples") 
        print(f"   Test: {len(self.test_dataset):,} samples")
    
    def _verify_label_distribution(self) -> None:
        """Verify label distribution across splits."""
        print(f"\n Label Distribution Analysis:")
        
        if self.dataset:
            splits_to_check = [
                ("Original Train", self.dataset["train"]),
                ("Original Test", self.dataset["test"])
            ]
        else:
            splits_to_check = [
                ("Train", self.train_dataset),
                ("Validation", self.val_dataset), 
                ("Test", self.test_dataset)
            ]
        
        for split_name, dataset_split in splits_to_check:
            if dataset_split:
                labels = dataset_split["label"]
                unique, counts = np.unique(labels, return_counts=True)
                
                print(f"   {split_name}:")
                for label, count in zip(unique, counts):
                    percentage = (count / len(labels)) * 100
                    label_name = self.config['model']['id2label'][label]
                    print(f"     {label_name}: {count:,} samples ({percentage:.1f}%)")
    
    def _print_dataset_structure(self) -> None:
        """Print detailed dataset structure."""
        print(f"\n Dataset Structure:")
        
        for split_name, dataset_split in [
            ("Train", self.train_dataset),
            ("Validation", self.val_dataset),
            ("Test", self.test_dataset)
        ]:
            if dataset_split:
                print(f"   {split_name}:")
                print(f"     Features: {list(dataset_split.features.keys())}")
                print(f"     Size: {len(dataset_split):,} samples")
                
                # Check first sample structure
                if len(dataset_split) > 0:
                    sample = dataset_split[0]
                    print(f"     Sample keys: {list(sample.keys())}")
                    if 'pixel_values' in sample:
                        pixel_vals = sample['pixel_values']
                        if hasattr(pixel_vals, 'shape'):
                            print(f"     Pixel values shape: {pixel_vals.shape}")
                        else:
                            print(f"     Pixel values type: {type(pixel_vals)}")
                            if isinstance(pixel_vals, list) and len(pixel_vals) > 0:
                                print(f"     Pixel values list length: {len(pixel_vals)}")
                                if hasattr(pixel_vals[0], 'shape'):
                                    print(f"     First item shape: {pixel_vals[0].shape}")
                                else:
                                    print(f"     First item type: {type(pixel_vals[0])}")
    
    def _check_sample_quality(self) -> None:
        """Check sample quality and preprocessing."""
        print(f"\n Sample Quality Check:")
        
        if self.train_dataset and len(self.train_dataset) > 0:
            sample = self.train_dataset[0]
            print(f"   Sample contains: {list(sample.keys())}")
            
            if 'image' in sample:
                print(f"   Original image size: {sample['image'].size}")
            if 'pixel_values' in sample:
                pixel_vals = sample['pixel_values']
                if hasattr(pixel_vals, 'shape'):
                    print(f"   Processed tensor shape: {pixel_vals.shape}")
                else:
                    print(f"   Pixel values type: {type(pixel_vals)}")
                    if isinstance(pixel_vals, list) and len(pixel_vals) > 0 and hasattr(pixel_vals[0], 'shape'):
                        print(f"   First tensor shape: {pixel_vals[0].shape}")
            
            print(f"   Label: {sample['label']} ({self.config['model']['id2label'][sample['label']]})")
    
    def visualize_samples(self, num_samples: int = 5) -> None:
        """
        Visualize random samples from the dataset.
        
        Args:
            num_samples: Number of samples to visualize
        """
        if not self.train_dataset:
            print(" Dataset not loaded. Please run the pipeline first.")
            return
        
        print(f"\n Visualizing {num_samples} random samples...")
        
        # Select random indices
        indices = np.random.choice(len(self.train_dataset), num_samples, replace=False)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        
        for i, idx in enumerate(indices):
            sample = self.train_dataset[idx]
            
            # Display original image if available, otherwise use processed
            if 'image' in sample:
                image = sample['image']
                axes[i].imshow(image, cmap='gray')
            elif 'pixel_values' in sample:
                # Convert tensor back to image for visualization
                image_array = sample['pixel_values'].numpy().transpose(1, 2, 0)
                axes[i].imshow(image_array, cmap='gray')
            
            label = sample['label']
            label_name = self.config['model']['id2label'][label]
            
            axes[i].set_title(f'Label: {label} ({label_name})')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

    def save_preprocessed_datasets(self, train_dataset: Dataset, 
                                val_dataset: Dataset, 
                                test_dataset: Dataset,
                                save_dir: str = "data/processed") -> None:
        """
        Save preprocessed datasets to disk so they can be reloaded later without preprocessing.

        Args:
            train_dataset (Dataset): Preprocessed training dataset
            val_dataset (Dataset): Preprocessed validation dataset
            test_dataset (Dataset): Preprocessed test dataset
            save_dir (str): Directory path to save datasets
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True,exist_ok=True)

        # Save each split
        train_dataset.save_to_disk(save_path / "train")
        val_dataset.save_to_disk(save_path / "val")
        test_dataset.save_to_disk(save_path / "test")

        print(f" Preprocessed datasets saved to disk at '{save_path.resolve()}'")

    
    def run_pipeline(self) -> Tuple[Any, Any, Any]:
        """
        Execute the complete dataset processing pipeline.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        print(" Starting dataset processing pipeline...")
        
        self.load_dataset()
        self.transform_labels()
        self.create_splits()
        self.setup_processor()
        self.preprocess_images()
        self.verify_dataset()
        self.save_preprocessed_datasets(self.train_dataset, self.val_dataset, self.test_dataset)


        print(" Dataset pipeline completed successfully!")
        
        return self.train_dataset, self.val_dataset, self.test_dataset

# Usage example
if __name__ == "__main__":
    # Initialize dataset manager
    dataset_manager = MNISTEvenOddDatasetManager("config/dataset.yaml")
    
    # Run complete pipeline
    train_ds, val_ds, test_ds = dataset_manager.run_pipeline()
    
    # Visualize samples
    dataset_manager.visualize_samples(3)