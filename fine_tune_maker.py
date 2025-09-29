
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models.moco_model import ModelMoCo
from transformers import AutoProcessor
from models.moco_model import create_moco_model

def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def ade_palette():
    return [[  0,   0,   0], [  0, 255,   0], [127, 255,   0], [  0, 127,   0],
            [255, 255,   0], [255,   0,   0], [127, 127,   0], [127,   0,   0]]

def get_model_with_pretrain_weight(model, weight_path):
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model.encoder_q

def get_input_image(PathDF, get_id, processor):
    img_path = PathDF['images'].iloc[get_id]
    image = Image.open(img_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")

    return inputs["pixel_values"].cuda(), image, img_path.split('/')[-1]

def get_color_seg(logits, image):
    # First, rescale logits to original image size
    logits = nn.functional.interpolate(logits.detach().cpu(),
                    size=image.size[::-1], # (height, width)
                    mode='bilinear',
                    align_corners=False)

    # Second, apply argmax on the class dimension
    seg = logits.argmax(dim=1)[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    return color_seg

def plot(image, color_seg, text):
    img = np.array(image) * 0.95 + color_seg * 0.05
    img = img.astype(np.uint8)

    return img

def path_transform(path):
    path_ = ''
    for str in path:
        if str == '\\':
            str = str.replace(str, '/')
        path_ += str
    return path_


def load_image_paths(image_directory):
    """
    Load all image paths from the specified directory and create a DataFrame.
    
    Args:
        image_directory (str): Path to the directory containing images
    
    Returns:
        pd.DataFrame: DataFrame with image paths
    """
    images = []
    for filename in os.listdir(image_directory):
        images.append(os.path.join(image_directory, filename))
    
    return pd.DataFrame({'images': images})


def setup_model_and_processor(weights_path):
    """
    Initialize the MoCo model and processor.
    
    Args:
        weights_path (str): Path to the model weights
    
    Returns:
        tuple: (model, processor)
    """
    moco_model = create_moco_model()
    model = get_model_with_pretrain_weight(moco_model, weights_path)
    processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")
    
    return model, processor


def setup_output_directory(input_path, output_folder_name='step_set_segmix_test'):
    """
    Create and validate the output directory.
    
    Args:
        input_path (str): Input directory path
        output_folder_name (str): Name of the output folder
    
    Returns:
        str: Output directory path
    """
    parent_dir = os.path.join(input_path, "..")
    output_path = os.path.join(parent_dir, output_folder_name)
    checkpath(output_path)
    
    if not os.path.exists(output_path):
        print(f'Warning: Path does not exist: {output_path}')
    if not os.access(output_path, os.W_OK):
        print(f'Warning: Path is not writable: {output_path}')
    
    return output_path


def process_single_image(PathDF, image_id, model, processor, output_path):
    """
    Process a single image and save the result.
    
    Args:
        PathDF (pd.DataFrame): DataFrame containing image paths
        image_id (int): Index of the image to process
        model: The trained model
        processor: Image processor
        output_path (str): Directory to save the output
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get image input
        input_tensor, image, filename = get_input_image(PathDF, image_id, processor)
        
        # Process with model
        _, logits = model(input_tensor)
        color_seg = get_color_seg(logits, image)
        
        # Create visualization
        result_image = plot(image, color_seg, 'trained')
        
        # Save image
        clean_filename = filename.split('\\')[-1]
        output_file_path = os.path.join(output_path, clean_filename)
        
        success = cv2.imwrite(output_file_path, np.array(result_image).astype(np.uint8))
        
        if success:
            print(f"Successfully saved: {clean_filename}")
        else:
            print(f"Failed to save: {clean_filename}")
        
        return success
        
    except Exception as e:
        print(f"Error processing image {image_id}: {str(e)}")
        return False


def process_all_images(PathDF, model, processor, output_path):
    """
    Process all images in the DataFrame.
    
    Args:
        PathDF (pd.DataFrame): DataFrame containing image paths
        model: The trained model
        processor: Image processor
        output_path (str): Directory to save outputs
    """
    total_images = len(PathDF)
    successful_count = 0
    
    print(f"Processing {total_images} images...")
    
    for image_id in range(total_images):
        print(f"Processing image {image_id + 1}/{total_images}")
        
        if process_single_image(PathDF, image_id, model, processor, output_path):
            successful_count += 1
    
    print(f"\nProcessing completed: {successful_count}/{total_images} images processed successfully")


def main():
    """
    Main function to orchestrate the image processing pipeline.
    """
    # Configuration
    trainimage_path = "image_folder"
    weights_path = "moco_model_path"
    
    try:
        # Load image paths
        print("Loading image paths...")
        PathDF = load_image_paths(trainimage_path)
        print(f"Found {len(PathDF)} images")
        
        # Setup model and processor
        print("Loading model and processor...")
        model, processor = setup_model_and_processor(weights_path)
        
        # Setup output directory
        print("Setting up output directory...")
        output_path = setup_output_directory(trainimage_path)
        
        # Process all images
        process_all_images(PathDF, model, processor, output_path)
        
        print('\n Successfully Completed! \n')
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()