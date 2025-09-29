"""
TripletMaskBinarization Three-Stage Processing Visualization Script

This script demonstrates the full process of TripletMaskBinarization 
three-stage processing on a pneumothorax sample image, including:
1. Original image and ground truth mask
2. Model probability prediction map
3. Three-stage process with triplet parameters [0.75, 1000, 0.3]:
   - Step 1: High threshold (0.75) region confirmation
   - Step 2: Area verification (>=1000 pixels)
   - Step 3: Low threshold (0.3) to generate final mask
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image

def get_DeepLabV3Plus(
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    deeplabv3_model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    # print(get_trainable_params(deeplabv3_model))
    return deeplabv3_model

def load_model_and_predict(image, model_path=None):
    """Load trained model and generate probability prediction map"""
    if model_path is None or not os.path.exists(model_path):
        print("Model path not provided or file doesn't exist, using simulated probability map...")
        return simulate_probability_map_fallback(image)
    
    try:
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize model architecture
        model = get_DeepLabV3Plus()
        
        # Load the trained weights with detailed error handling
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        # Debug: Print checkpoint structure
        print(f"Checkpoint type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                print("Loading from 'model_state_dict'")
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                print("Loading from 'state_dict'")
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                print("Loading from 'model'")
                state_dict = checkpoint['model']
            else:
                print("Loading checkpoint as state_dict directly")
                state_dict = checkpoint
        else:
            # Direct state dict
            print("Loading checkpoint as direct state_dict")
            state_dict = checkpoint
        
        # Debug: Print some state_dict keys
        if isinstance(state_dict, dict):
            print(f"State dict keys (first 5): {list(state_dict.keys())[:5]}")
        
        # Debug: Check what keys the model expects
        model_keys = list(model.state_dict().keys())
        print(f"Model expects keys (first 5): {model_keys[:5]}")
        
        # Check if model expects 'model.' prefix
        model_expects_model_prefix = model_keys and model_keys[0].startswith('model.')
        checkpoint_has_model_prefix = any(key.startswith('model.') for key in state_dict.keys())
        
        print(f"Model expects 'model.' prefix: {model_expects_model_prefix}")
        print(f"Checkpoint has 'model.' prefix: {checkpoint_has_model_prefix}")
        
        final_state_dict = state_dict
        
        # Handle prefix mismatch
        if model_expects_model_prefix and not checkpoint_has_model_prefix:
            # Model expects 'model.' but checkpoint doesn't have it - add prefix
            print("Adding 'model.' prefix to checkpoint keys")
            final_state_dict = {}
            for key, value in state_dict.items():
                new_key = f'model.{key}' if not key.startswith('model.') else key
                final_state_dict[new_key] = value
        elif not model_expects_model_prefix and checkpoint_has_model_prefix:
            # Model doesn't expect 'model.' but checkpoint has it - remove prefix
            print("Removing 'model.' prefix from checkpoint keys")
            final_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('model.', '') if key.startswith('model.') else key
                final_state_dict[new_key] = value
        else:
            # Prefixes match - use as-is
            print("Key prefixes match - using checkpoint as-is")
            final_state_dict = state_dict
        
        # Handle 'module.' prefix (from DataParallel) if present
        if any(key.startswith('module.') for key in final_state_dict.keys()):
            print("Removing 'module.' prefix from checkpoint keys")
            temp_state_dict = {}
            for key, value in final_state_dict.items():
                new_key = key.replace('module.', '') if key.startswith('module.') else key
                temp_state_dict[new_key] = value
            final_state_dict = temp_state_dict
        
        print(f"Final state dict keys (first 5): {list(final_state_dict.keys())[:5]}")
        
        # Try to load state dict
        try:
            missing_keys, unexpected_keys = model.load_state_dict(final_state_dict, strict=False)
            
            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                print("Model state dict loaded perfectly!")
            else:
                if missing_keys:
                    print(f"Warning: Missing keys ({len(missing_keys)}): {missing_keys[:3]}...")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:3]}...")
                print("Model state dict loaded with warnings")
        except Exception as load_error:
            print(f"Error loading state dict: {load_error}")
            raise load_error
        
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        
        # Prepare image for model input
        transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((768, 768)),  # Match your training size
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
            
        # Transform image
        input_tensor = transform(image_rgb).unsqueeze(0).to(device)
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Generate prediction
        with torch.no_grad():
            output = model(input_tensor)
            print(f"Model output shape: {output.shape}")
            
            # Apply sigmoid for binary segmentation
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
            
            # Resize back to original image size
            prob_map = cv2.resize(prob_map, (image.shape[1], image.shape[0]))
            
        print(f"Model prediction completed. Probability range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")
        return prob_map
        
    except Exception as e:
        print(f"Error loading model or generating prediction: {e}")
        import traceback
        traceback.print_exc()  # Print full error traceback
        print("Falling back to simulated probability map...")
        return simulate_probability_map_fallback(image)

def simulate_probability_map_fallback(image):
    """Fallback function to simulate probability map if model loading fails"""
    print("Generating simulated probability map...")
    # Create a simple simulated probability map
    prob_map = np.random.uniform(0.0, 0.2, image.shape).astype(np.float32)
    
    # Add some higher probability regions randomly
    h, w = image.shape
    num_regions = np.random.randint(1, 3)
    
    for _ in range(num_regions):
        center_y = np.random.randint(h//4, 3*h//4)
        center_x = np.random.randint(w//4, 3*w//4)
        radius = np.random.randint(30, 80)
        
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        prob_map[mask] = np.random.uniform(0.6, 0.9, np.sum(mask))
    
    return prob_map

def load_sample_with_pneumothorax():
    """Load a sample image with pneumothorax"""
    # Choose a training image with pneumothorax (train_1 means pneumothorax present)
    image_files_with_pneumo = [
        "7_test_1__SPD_C1992_T1.png"
    ]
    
    # Select the first available image
    base_path = os.getcwd()  # Current working directory
    
    for img_file in image_files_with_pneumo:
        img_path = f"{base_path}/image/{img_file}"
        mask_path = f"{base_path}/mask/{img_file}"
        
        try:
            # Load image (grayscale)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (768, 768))
            mask = cv2.resize(mask, (768, 768))
            if image is not None and mask is not None:
                # Check if mask actually has pneumothorax (nonzero pixels)
                if np.any(mask > 0):
                    print(f"Selected image: {img_file}")
                    print(f"Image shape: {image.shape}")
                    print(f"Number of pneumothorax pixels in mask: {np.sum(mask > 0)}")
                    return image, mask, img_file
        except Exception as e:
            print(f"Cannot load {img_file}: {e}")
            continue
    
    return None, None, None

def apply_triplet_binarization(prob_map, top_threshold=0.75, area_threshold=1000, bottom_threshold=0.3):
    """Apply triplet mask binarization algorithm"""
    # Step 1: High threshold mask (confirmed region)
    high_confidence_mask = (prob_map > top_threshold).astype(np.uint8)
    
    # Step 2: Area verification
    high_confidence_area = np.sum(high_confidence_mask)
    
    # Step 3: Generate final mask
    if high_confidence_area >= area_threshold:
        # Area is sufficient, use low threshold to generate mask
        final_mask = (prob_map > bottom_threshold).astype(np.uint8)
        passed_area_check = True
    else:
        # Area is insufficient, reject the entire prediction
        final_mask = np.zeros_like(prob_map, dtype=np.uint8)
        passed_area_check = False
    
    return high_confidence_mask, high_confidence_area, final_mask, passed_area_check

def create_visualization(model_path=None):
    """Create the full visualization"""
    # Load sample data
    image, true_mask, filename = load_sample_with_pneumothorax()
    
    if image is None:
        print("Could not find a suitable pneumothorax sample image")
        return
    
    # Generate probability prediction using model or fallback
    prob_map = load_model_and_predict(image, model_path)
    print(f"Probability map shape: {prob_map.shape}, min: {prob_map.min()}, max: {prob_map.max()}"
          )
    # Apply triplet binarization [0.75, 750, 0.3]
    high_mask, high_area, final_mask, passed = apply_triplet_binarization(
        prob_map, top_threshold=0.75, area_threshold=750, bottom_threshold=0.3
    )
    
    # Create plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    model_info = f"Model: {os.path.basename(model_path)}" if model_path and os.path.exists(model_path) else "Simulated"
    fig.suptitle(f'TripletMaskBinarization Three-Stage Visualization\nSample: {filename} | {model_info} | Params: [0.75, 1000, 0.3]', 
                 fontsize=16, fontweight='bold')
    
    # First row: input data
    # 1. Original image
    axes[0,0].imshow(image, cmap='gray')
    axes[0,0].set_title('1. Original Chest X-ray', fontsize=12, fontweight='bold')
    axes[0,0].axis('off')
    
    # 2. Ground truth mask
    axes[0,1].imshow(true_mask, cmap='Reds', alpha=0.8)
    axes[0,1].imshow(image, cmap='gray', alpha=0.3)
    axes[0,1].set_title('2. Ground Truth Mask', fontsize=12, fontweight='bold')
    axes[0,1].axis('off')
    
    # 3. Probability prediction map
    im3 = axes[0,2].imshow(prob_map, cmap='viridis', vmin=0, vmax=1)
    prediction_type = "Model Prediction" if model_path and os.path.exists(model_path) else "Simulated Prediction"
    axes[0,2].set_title(f'3. {prediction_type}\nProbability Output', fontsize=12, fontweight='bold')
    axes[0,2].axis('off')
    plt.colorbar(im3, ax=axes[0,2], fraction=0.046, pad=0.04)
    
    # 4. Probability histogram with thresholds
    axes[0,3].hist(prob_map.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,3].axvline(x=0.75, color='red', linestyle='--', linewidth=2, label='High Threshold (0.75)')
    axes[0,3].axvline(x=0.3, color='orange', linestyle='--', linewidth=2, label='Low Threshold (0.3)')
    axes[0,3].set_title('4. Probability Distribution', fontsize=12, fontweight='bold')
    axes[0,3].set_xlabel('Probability')
    axes[0,3].set_ylabel('Pixel Count')
    axes[0,3].legend()
    axes[0,3].grid(True, alpha=0.3)
    
    # Second row: three-stage process
    # Step 1: High threshold confirmation
    axes[1,0].imshow(high_mask, cmap='Reds')
    axes[1,0].set_title(f'Step 1: High Threshold\n(> 0.75)\nDetected {high_area} pixels', 
                       fontsize=12, fontweight='bold')
    axes[1,0].axis('off')
    
    # Step 2: Area verification
    if passed:
        color = 'green'
        result_text = f'Passed\n{high_area} >= 1000'
    else:
        color = 'red'
        result_text = f'Not Passed\n{high_area} < 1000'
    
    axes[1,1].imshow(high_mask, cmap='Reds')
    axes[1,1].add_patch(Rectangle((10, 10), image.shape[1]-20, image.shape[0]-20, 
                                 linewidth=5, edgecolor=color, facecolor='none'))
    axes[1,1].set_title(f'Step 2: Area Verification\n(>= 1000 pixels)\n{result_text}', 
                       fontsize=12, fontweight='bold')
    axes[1,1].axis('off')
    
    # Step 3: Final mask generation
    if passed:
        axes[1,2].imshow(final_mask, cmap='Greens')
        title_text = 'Step 3: Final Mask\n(> 0.3)\nMask Generated'
    else:
        axes[1,2].imshow(final_mask, cmap='Greys')
        title_text = 'Step 3: Final Mask\n(> 0.3)\nRejected'
    
    axes[1,2].set_title(title_text, fontsize=12, fontweight='bold')
    axes[1,2].axis('off')
    
    # Result comparison
    axes[1,3].imshow(image, cmap='gray', alpha=0.7)
    axes[1,3].imshow(final_mask, cmap='Greens', alpha=0.6)
    axes[1,3].imshow(true_mask, cmap='Reds', alpha=0.3)
    axes[1,3].set_title('4. Result Comparison\nGreen: Prediction  Red: Ground Truth', fontsize=12, fontweight='bold')
    axes[1,3].axis('off')
    
    # Add legend
    green_patch = mpatches.Patch(color='green', alpha=0.6, label='TripletMask Prediction')
    red_patch = mpatches.Patch(color='red', alpha=0.3, label='Ground Truth Mask')
    axes[1,3].legend(handles=[green_patch, red_patch], loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
#     # Add algorithm description
#     algorithm_text = '''
# TripletMaskBinarization Algorithm Logic:
# 1. High-confidence detection (top_score_threshold=0.75): Find regions with definite pneumothorax
# 2. Area verification (area_threshold=1000): Check if high-confidence region meets minimum area requirement
# 3. Mask generation (bottom_score_threshold=0.3): If passed, use low threshold to generate full mask
   
# Medical logic: Avoid mistaking small noisy regions for pneumothorax, while preserving the full boundary of real pneumothorax
#     '''
    
    # fig.text(0.02, 0.02, algorithm_text, fontsize=10, 
    #          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Save image
    output_path = "./triplet_visualization_3.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Show statistics
    print(f"\n=== Processing Statistics ===")
    print(f"Sample file: {filename}")
    print(f"Prediction method: {'Model' if model_path and os.path.exists(model_path) else 'Simulated'}")
    print(f"Number of pneumothorax pixels (ground truth): {np.sum(true_mask > 0)}")
    print(f"Number of high-confidence region pixels: {high_area}")
    print(f"Area verification result: {'Passed' if passed else 'Not Passed'}")
    print(f"Number of final predicted pixels: {np.sum(final_mask > 0)}")
    
    if np.sum(true_mask > 0) > 0 and np.sum(final_mask > 0) > 0:
        # Calculate simple overlap
        intersection = np.sum((true_mask > 0) & (final_mask > 0))
        union = np.sum((true_mask > 0) | (final_mask > 0))
        iou = intersection / union if union > 0 else 0
        print(f"IoU (Intersection over Union): {iou:.3f}")
    
    plt.show()

if __name__ == "__main__":
    # Example usage with model path
    model_path = "./deeplabv3plus_768_fold3_epoch21.pth"  # Replace with your actual model path
    # model_path = None  # Use None for simulated prediction
    
    create_visualization(model_path)