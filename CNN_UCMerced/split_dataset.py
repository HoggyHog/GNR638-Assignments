import os
import json
import shutil
import random

def organize_ucmerced_from_json(dataset_path, json_path):
    """
    Organizes UC Merced dataset into train, val, and test folders using split_data.json.

    Args:
        dataset_path (str): Path to the dataset (contains all class folders).
        json_path (str): Path to the split_data.json file.
    """
    
    # Load split information
    with open(json_path, "r") as f:
        split_data = json.load(f)

    output_dirs = {
        "train": os.path.join(dataset_path, "train"),
        "val": os.path.join(dataset_path, "val"),
        "test": os.path.join(dataset_path, "test"),
    }

    # Create train, val, test directories
    for split in output_dirs.values():
        os.makedirs(split, exist_ok=True)

    # Move images based on split_data.json
    missing_count = 0
    moved_count = 0

    for split, images in split_data.items():
        for img_info in images:  
            img_path = img_info[0]  # Extract actual image path
            class_name = img_info[2]  # Extract class name

            src_path = os.path.join(dataset_path, img_path)  # Full path to source image
            dest_dir = os.path.join(output_dirs[split], class_name)  # Target directory

            # Create class folder in destination if not exists
            os.makedirs(dest_dir, exist_ok=True)

            # Move image if it exists
            if os.path.exists(src_path):
                shutil.move(src_path, os.path.join(dest_dir, os.path.basename(img_path)))
                moved_count += 1
            else:
                print(f"⚠️ Warning: {src_path} not found!")
                missing_count += 1

    print(f"✅ Dataset organized! Moved: {moved_count} images, Missing: {missing_count}")



def split_dataset(dataset_path, train_ratio, val_ratio, test_ratio):
    """
    Splits UC Merced dataset into train, val, and test folders.

    Args:
        dataset_path (str): Path to the dataset (contains class folders).
        train_ratio (float): Percentage of images for training.
        val_ratio (float): Percentage of images for validation.
        test_ratio (float): Percentage of images for testing.
    """

    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"

    # Define output directories
    output_dirs = {
        "train": os.path.join(dataset_path, "train"),
        "val": os.path.join(dataset_path, "val"),
        "test": os.path.join(dataset_path, "test"),
    }

    # Create train, val, test directories
    for split in output_dirs.values():
        os.makedirs(split, exist_ok=True)

    # Get all class folders
    class_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    for class_name in class_folders:
        class_path = os.path.join(dataset_path, class_name)
        images = [img for img in os.listdir(class_path) if img.endswith(('.jpg', '.png','tif'))]

        random.shuffle(images)

        train_split = int(len(images) * train_ratio)
        val_split = int(len(images) * (train_ratio + val_ratio))

        subsets = {
            "train": images[:train_split],
            "val": images[train_split:val_split],
            "test": images[val_split:]
        }

        for split, imgs in subsets.items():
            split_dir = os.path.join(output_dirs[split], class_name)
            os.makedirs(split_dir, exist_ok=True)
            for img in imgs:
                shutil.move(os.path.join(class_path, img), os.path.join(split_dir, img))

        # Remove the original class folder if empty
        if not os.listdir(class_path):
            os.rmdir(class_path)

    print("✅ Dataset successfully split into train, val, and test folders!")



import shutil

# Remove old train/val/test folders
dataset_path = "/Users/sripradheep/Downloads/CNN_UCMerced/Ucmerced/Images"

# Now re-split dataset
split_dataset(dataset_path, train_ratio=0.5, val_ratio=0.1, test_ratio=0.4)


