# data_loader.py
from datasets import load_dataset
import json
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, dataset_name="katanaml-org/invoices-donut-data-v1"):
        self.dataset_name = dataset_name
        self.dataset = None
    
    def load_dataset(self):
        """Load the invoices dataset from Hugging Face"""
        print("📂 Loading dataset...")
        try:
            self.dataset = load_dataset(self.dataset_name)
            print(f"✅ Dataset loaded successfully!")
            print(f"   Split: {list(self.dataset.keys())}")
            print(f"   Training samples: {len(self.dataset['train'])}")
            return self.dataset
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def get_sample(self, index=0):
        """Get a specific sample from the dataset"""
        if self.dataset is None:
            if not self.load_dataset():
                return None
        return self.dataset['train'][index]
    
    def get_sample_info(self, index=0):
        """Get information about a sample"""
        sample = self.get_sample(index)
        if sample is None:
            return None
            
        print(f"\n📄 Sample {index} Information:")
        print(f"   Image type: {type(sample['image'])}")
        print(f"   Image size: {sample['image'].size}")
        print(f"   Keys available: {list(sample.keys())}")
        
        # Show ground truth structure
        try:
            ground_truth = json.loads(sample['ground_truth'])
            print(f"   Ground truth fields: {list(ground_truth.keys())}")
            return sample, ground_truth
        except json.JSONDecodeError as e:
            print(f"   Error parsing ground truth: {e}")
            return sample, None
    
    def display_sample_image(self, index=0, save_path=None):
        """Display a sample invoice image"""
        sample = self.get_sample(index)
        if sample is None:
            return
            
        plt.figure(figsize=(10, 14))
        plt.imshow(sample['image'])
        plt.title(f"Invoice Sample {index}")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"✅ Image saved to {save_path}")
        
        plt.show()

# Test the data loader
if __name__ == "__main__":
    print("🧪 Testing DataLoader...")
    
    loader = DataLoader()
    
    # Test dataset loading
    dataset = loader.load_dataset()
    
    if dataset is not None:
        # Test sample access
        sample, ground_truth = loader.get_sample_info(0)
        
        if sample and ground_truth:
            print("\n📋 Sample ground truth data:")
            for key, value in list(ground_truth.items())[:3]:  # Show first 3 items
                print(f"   {key}: {value}")
            
            print("\n✅ DataLoader test completed successfully!")
        else:
            print("\n❌ Failed to load sample data!")
    else:
        print("\n❌ Failed to load dataset!")