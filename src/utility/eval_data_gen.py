import os
import json
import shutil
from pathlib import Path

# Configuration
IMAGE_SOURCE_DIR = "../../invoices_dataset/unified_dataset/images"
JSON_SOURCE_DIR = "../../invoices_dataset/unified_dataset/json"
OUTPUT_DIR = "../../invoices_dataset/eval_data"
NUM_SAMPLES = 200
PREFIX = "dataset1_katanaml_train_"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all matching image files and sort them
image_files = sorted([
    f for f in os.listdir(IMAGE_SOURCE_DIR)
    if f.startswith(PREFIX) and f.endswith(('.jpg', '.jpeg', '.png'))
])

# Select first 200 files
selected_files = image_files[:NUM_SAMPLES]

# Process each selected file
for image_file in selected_files:
    image_path = os.path.join(IMAGE_SOURCE_DIR, image_file)
    json_file = image_file.rsplit('.', 1)[0] + '.json'
    json_path = os.path.join(JSON_SOURCE_DIR, json_file)
    
    # Copy image
    shutil.copy2(image_path, os.path.join(OUTPUT_DIR, image_file))
    
    # Process JSON
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract fields from nested structure
        ground_truth = {}
        
        # Navigate through gt_parse structure
        if "gt_parse" in data:
            gt_parse = data["gt_parse"]
            
            # Extract from header
            if "header" in gt_parse:
                header = gt_parse["header"]
                if "invoice_no" in header:
                    ground_truth["invoice_nr"] = header["invoice_no"]
                if "invoice_date" in header:
                    ground_truth["date"] = header["invoice_date"]
            
            # Extract from summary
            if "summary" in gt_parse:
                summary = gt_parse["summary"]
                if "total_gross_worth" in summary:
                    total = summary["total_gross_worth"]
                    # Remove spaces
                    total = total.replace(" ", "")
                    # Remove dollar sign from beginning if present
                    if total.startswith("$"):
                        total = total[1:]
                    ground_truth["total_amount"] = total
        
        # Save reshaped JSON
        output_json_path = os.path.join(OUTPUT_DIR, json_file)
        with open(output_json_path, 'w') as f:
            json.dump(ground_truth, f, indent=2)
    else:
        print(f"Warning: JSON file not found for {image_file}")

print(f"Collected {len(selected_files)} invoices to {OUTPUT_DIR}")