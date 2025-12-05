import os
import hashlib
import shutil
from collections import defaultdict

def find_exact_duplicates_simple():
    """Simple MD5-based duplicate detection"""
    
    unified_dir = "./invoices_dataset/unified_dataset"
    images_dir = f"{unified_dir}/images"
    
    print("🔍 FINDING EXACT DUPLICATES (MD5 HASH)")
    print("=" * 50)
    
    # Calculate MD5 hashes for all images
    file_hashes = defaultdict(list)
    
    for img_file in os.listdir(images_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = f"{images_dir}/{img_file}"
            
            # Calculate MD5 hash
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            file_hashes[file_hash].append(img_file)
    
    # Find duplicates
    duplicate_groups = []
    for file_hash, files in file_hashes.items():
        if len(files) > 1:
            duplicate_groups.append(files)
    
    print(f"📊 Found {len(duplicate_groups)} duplicate groups")
    print(f"📊 Total duplicate files: {sum(len(group) - 1 for group in duplicate_groups)}")
    
    # Show examples
    for i, group in enumerate(duplicate_groups[:5]):
        print(f"\n🔍 Group {i+1}:")
        print(f"   Keep: {group[0]}")
        for duplicate in group[1:]:
            print(f"   Remove: {duplicate}")
    
    return duplicate_groups

def remove_duplicates_simple(duplicate_groups):
    """Remove duplicates found by MD5 hashing"""
    
    unified_dir = "./invoices_dataset/unified_dataset"
    images_dir = f"{unified_dir}/images"
    json_dir = f"{unified_dir}/json"
    
    files_to_remove = set()
    for group in duplicate_groups:
        files_to_remove.update(group[1:])  # Remove all but first file
    
    print(f"\n🗑️  Removing {len(files_to_remove)} duplicate files...")
    
    # Create clean directory
    temp_dir = f"{unified_dir}_clean"
    os.makedirs(f"{temp_dir}/images", exist_ok=True)
    os.makedirs(f"{temp_dir}/json", exist_ok=True)
    
    # Copy non-duplicate files
    for img_file in os.listdir(images_dir):
        if img_file not in files_to_remove:
            shutil.copy2(f"{images_dir}/{img_file}", f"{temp_dir}/images/{img_file}")
            
            # Copy corresponding JSON
            json_file = os.path.splitext(img_file)[0] + '.json'
            if os.path.exists(f"{json_dir}/{json_file}"):
                shutil.copy2(f"{json_dir}/{json_file}", f"{temp_dir}/json/{json_file}")
    
    # Replace directory
    shutil.rmtree(unified_dir)
    shutil.move(temp_dir, unified_dir)
    
    return len(files_to_remove)

# Run simple version
if __name__ == "__main__":
    duplicates = find_exact_duplicates_simple()
    
    if duplicates:
        response = input(f"\nRemove {sum(len(group)-1 for group in duplicates)} duplicates? (y/N): ")
        if response.lower() == 'y':
            removed_count = remove_duplicates_simple(duplicates)
            print(f"✅ Removed {removed_count} duplicate files")
        else:
            print("❌ Operation cancelled")
    else:
        print("🎉 No exact duplicates found!")