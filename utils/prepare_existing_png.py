import os
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil
import cv2


def prepare_existing_png_data(src_image_dir, src_mask_dir, dst_train_dir, dst_mask_dir, dst_test_dir):
    """
    Process existing PNG image data, organizing it into the structure required for training.
    based on the train/test tags in the filenames.
    
    Args:
        src_image_dir: original image directory (siim-acr-pneumothorax/png_images)
        src_mask_dir: original mask directory (siim-acr-pneumothorax/png_masks)
        dst_train_dir: target training image directory
        dst_mask_dir: target mask directory
        dst_test_dir: target testing image directory
    """

    # Ensure target directories exist
    os.makedirs(dst_train_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)
    os.makedirs(dst_test_dir, exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir(src_image_dir) if f.endswith('.png')]
    
    print(f"Found {len(image_files)} PNG images")
    
    train_count = 0
    test_count = 0
    
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(src_image_dir, img_file)

        # Determine if the image is for training or testing based on the filename
        # Filename format: {id}_train_{label}_.png or {id}_test_{label}_.png
        if '_train_' in img_file:
            # Training image
            dst_img_path = os.path.join(dst_train_dir, img_file)
            shutil.copy2(img_path, dst_img_path)

            # Copy corresponding mask if it exists
            mask_path = os.path.join(src_mask_dir, img_file)
            if os.path.exists(mask_path):
                dst_mask_path = os.path.join(dst_mask_dir, img_file)
                shutil.copy2(mask_path, dst_mask_path)
            
            train_count += 1
            
        elif '_test_' in img_file:
            # Testing image
            dst_img_path = os.path.join(dst_test_dir, img_file)
            shutil.copy2(img_path, dst_img_path)
            test_count += 1

    print(f"Processed {train_count} training images and {test_count} testing images")


def create_folds_csv(src_image_dir, src_mask_dir, output_csv_path, n_folds=5):
    """
    Create folds CSV file based on existing PNG files
    
    Args:
        src_image_dir: original image directory
        src_mask_dir: original mask directory
        output_csv_path: output CSV file path
        n_folds: number of folds
    """

    # Get all training image files
    image_files = [f for f in os.listdir(src_image_dir) if f.endswith('.png') and '_train_' in f]
    
    print(f"Creating folds for {len(image_files)} training images")
    
    data = []
    
    for img_file in tqdm(image_files, desc="Analyzing images"):
        # Check if the corresponding mask exists and is not empty
        mask_path = os.path.join(src_mask_dir, img_file)
        exist_labels = 0
        
        if os.path.exists(mask_path):
            # Load mask and check for labels
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # If there are non-zero pixels in the mask, it is considered to have labels
                if np.any(mask > 0):
                    exist_labels = 1

        # Assign fold based on labels in filename
        # Extract ID part from filename for fold assignment
        parts = img_file.split('_')
        if len(parts) >= 3:
            try:
                image_id = int(parts[0])
                fold = image_id % n_folds
            except:
                fold = hash(img_file) % n_folds
        else:
            fold = hash(img_file) % n_folds
        
        data.append({
            'fname': img_file,
            'fold': fold,
            'exist_labels': exist_labels
        })

    # Create DataFrame and save
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    
    print(f"Created folds CSV with {len(df)} entries")
    print(f"Positive samples: {df[df.exist_labels == 1].shape[0]}")
    print(f"Negative samples: {df[df.exist_labels == 0].shape[0]}")

    # Show distribution of each fold
    print("\nFold distribution:")
    for fold in range(n_folds):
        fold_data = df[df.fold == fold]
        pos_count = fold_data[fold_data.exist_labels == 1].shape[0]
        neg_count = fold_data[fold_data.exist_labels == 0].shape[0]
        print(f"Fold {fold}: {pos_count} positive, {neg_count} negative (total: {len(fold_data)})")
    
    return df


# def update_dataset_paths(dataset_file_path, new_data_folder):
#     """
#     Update dataset paths in Pneumadataset.py
#     """
#     if not os.path.exists(dataset_file_path):
#         print(f"Dataset file {dataset_file_path} not found")
#         return
    
#     with open(dataset_file_path, 'r', encoding='utf-8') as f:
#         content = f.read()

#     # Update path settings
#     old_train_path = "self.train_image_path = '{}/train/'.format(data_folder)"
#     new_train_path = f"self.train_image_path = '{new_data_folder}/train/'"
    
#     old_mask_path = "self.train_mask_path = '{}/mask/'.format(data_folder)"
#     new_mask_path = f"self.train_mask_path = '{new_data_folder}/mask/'"
    
#     old_test_path = "self.test_image_path = '{}/test/'.format(data_folder)"
#     new_test_path = f"self.test_image_path = '{new_data_folder}/test/'"
    
#     content = content.replace(old_train_path, new_train_path)
#     content = content.replace(old_mask_path, new_mask_path)
#     content = content.replace(old_test_path, new_test_path)
    
#     with open(dataset_file_path, 'w', encoding='utf-8') as f:
#         f.write(content)
    
#     print(f"Updated dataset paths in {dataset_file_path}")


if __name__ == "__main__":
    # Set paths - based on your actual folder structure
    base_dir = "./siim-acr-pneumothorax"
    src_image_dir = f"{base_dir}/step_set_segmix_v7.0"
    print(os.listdir(base_dir))
    src_mask_dir = f"{base_dir}/png_masks"
    
    dst_train_dir = f"{base_dir}/step_new/train_step"
    dst_mask_dir = f"{base_dir}/step_new/mask_step"
    dst_test_dir = f"{base_dir}/step_new/test_step"

    # Step 1: Organizing PNG images
    print("Step 1: Organizing PNG images...")
    prepare_existing_png_data(src_image_dir, src_mask_dir, dst_train_dir, dst_mask_dir, dst_test_dir)

    # Step 2: Creating folds CSV
    print("\nStep 2: Creating folds CSV...")
    folds_csv_path = "./siim-acr-pneumothorax/folds/train_folds_5_png.csv"
    create_folds_csv(dst_train_dir, dst_mask_dir, folds_csv_path)
    
    print("\n Data preparation completed!")
    print(f" Training images: {dst_train_dir}")
    print(f" Training masks: {dst_mask_dir}")
    print(f" Test images: {dst_test_dir}")
    print(f" Folds CSV: {folds_csv_path}")
