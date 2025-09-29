import os

import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset


class PneumothoraxDataset(Dataset):
    """
    PneumothoraxDataset

    Data processing for pneumothorax segmentation training / validation / testing modes
    """
    
    def __init__(self, data_folder, mode, transform=None,
                 fold_index=None, folds_distr_path=None):
        """
        Initialize the dataset

        Args:
            data_folder: folder path
            mode: mode ('train', 'val', 'test')
            transform: data augmentation transforms
            fold_index: fold index
            folds_distr_path: fold distribution file path
        """
        self.transform = transform
        self.mode = mode

        # Set paths
        self.train_image_path = '{}/train_step/'.format(data_folder)
        self.train_mask_path = '{}/mask_step/'.format(data_folder)
        self.test_image_path = '{}/test_step/'.format(data_folder)

        self.fold_index = None
        self.folds_distr_path = folds_distr_path
        self.set_mode(mode, fold_index)

    def _numpy_to_tensor(self, array, is_image=True):
        """
        Convert numpy array to the correct format tensor

        Args:
            array: numpy array
            is_image: whether it is an image (affects dimension conversion)

        Returns:
            torch.Tensor: converted tensor
        """
        if not isinstance(array, np.ndarray):
            return array

        # Convert to float32 tensor
        if is_image and len(array.shape) == 3:  # Image HWC -> CHW
            tensor = torch.from_numpy(array.transpose(2, 0, 1)).float()
        elif not is_image and len(array.shape) == 2:  # Mask HW -> CHW
            tensor = torch.from_numpy(array).float().unsqueeze(0)
        else:
            tensor = torch.from_numpy(array).float()
            if len(tensor.shape) == 2:  # Ensure channel dimension exists
                tensor = tensor.unsqueeze(0)

        # Normalize to [0, 1] if values are in [0, 255] range
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
            
        return tensor

    def _load_image(self, image_path):
        """Load image"""
        return cv2.imread(image_path, 1)
    
    def _load_mask(self, mask_path, exist_label=True):
        """Load mask"""
        if not exist_label:
            return np.zeros((1024, 1024))
        return cv2.imread(mask_path, 0)
    
    def _apply_transform(self, image, mask=None):
        """Apply data augmentation transforms"""
        if not self.transform:
            return self._convert_to_tensor_no_transform(image, mask)
        
        if mask is not None:
            sample = {"image": image, "mask": mask}
            sample = self.transform(**sample)
            return self._convert_transformed_sample(sample)
        else:
            sample = {"image": image}
            sample = self.transform(**sample)
            return self._convert_transformed_image(sample['image'])
    
    def _convert_transformed_sample(self, sample):
        """Convert transformed sample"""
        image = self._numpy_to_tensor(sample['image'], is_image=True)
        mask = self._numpy_to_tensor(sample['mask'], is_image=False)
        return image, mask
    
    def _convert_transformed_image(self, image):
        """Convert transformed image"""
        return self._numpy_to_tensor(image, is_image=True)
    
    def _convert_to_tensor_no_transform(self, image, mask=None):
        """No transform tensor conversion"""
        image_tensor = self._numpy_to_tensor(image, is_image=True)
        
        if mask is not None:
            mask_tensor = self._numpy_to_tensor(mask, is_image=False)
            return image_tensor, mask_tensor
        
        return image_tensor

    def set_mode(self, mode, fold_index):
        """
        Set dataset mode and load corresponding data list

        Args:
            mode: mode ('train', 'val', 'test')
            fold_index: fold index
        """
        self.mode = mode
        self.fold_index = fold_index

        if self.mode in ['train', 'val']:
            self._setup_train_val_mode(fold_index)
        elif self.mode == 'test':
            self._setup_test_mode()
    
    def _setup_train_val_mode(self, fold_index):
        """Set up training or validation mode"""
        folds = pd.read_csv(self.folds_distr_path)
        folds.fold = folds.fold.astype(str)
        
        if self.mode == 'train':
            folds = folds[folds.fold != fold_index]
            self.train_list = folds.fname.values.tolist()
            self.exist_labels = folds.exist_labels.values.tolist()
        else:  # val mode
            folds = folds[folds.fold == fold_index]
            self.val_list = folds.fname.values.tolist()
            
        self.num_data = len(folds)
    
    def _setup_test_mode(self):
        """Set up test mode"""
        self.test_list = sorted(os.listdir(self.test_image_path))
        self.num_data = len(self.test_list)

    def __getitem__(self, index):
        if self.fold_index is None and self.mode != 'test':
            print('WRONG!!!!!!! fold index is NONE!!!!!!!!!!!!!!!!!')
            return
        
        if self.mode == 'test':
            return self._get_test_item(index)
        elif self.mode == 'train':
            return self._get_train_item(index)
        elif self.mode == 'val':
            return self._get_val_item(index)
    
    def _get_train_item(self, index):
        """Training data processing"""
        image_path = os.path.join(self.train_image_path, self.train_list[index])
        mask_path = os.path.join(self.train_mask_path, self.train_list[index])
        
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path, exist_label=bool(self.exist_labels[index]))
        
        return self._apply_transform(image, mask)
    
    def _get_val_item(self, index):
        """Validation data processing"""
        image_path = os.path.join(self.train_image_path, self.val_list[index])
        mask_path = os.path.join(self.train_mask_path, self.val_list[index])
        
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path, exist_label=True)
        
        return self._apply_transform(image, mask)
    
    def _get_test_item(self, index):
        """Testing data processing"""
        image_path = os.path.join(self.test_image_path, self.test_list[index])
        image = self._load_image(image_path)
        
        if self.transform:
            image = self._apply_transform(image)
        else:
            image = self._convert_to_tensor_no_transform(image)
            
        image_id = self.test_list[index].replace('.png', '')
        return image_id, image
    
    def __len__(self):
        return self.num_data


from torch.utils.data.sampler import Sampler

class PneumoSampler(Sampler):
    """
    Pneumothorax dataset sampler

    Control the ratio of positive and negative samples to ensure sufficient sampling frequency of pneumothorax images
    """
    
    def __init__(self, folds_distr_path, fold_index, demand_non_empty_proba):
        """
        Initialize sampler
        
        Args:
            folds_distr_path: fold distribution file path
            fold_index: fold index
            demand_non_empty_proba: non-empty mask (pneumothorax) sampling probability
        """
        assert demand_non_empty_proba > 0, 'frequency of non-empty images must be greater than zero'
        self.fold_index = fold_index
        self.positive_proba = demand_non_empty_proba

        # Load and filter data
        self.folds = pd.read_csv(folds_distr_path)
        self.folds.fold = self.folds.fold.astype(str)
        self.folds = self.folds[self.folds.fold != fold_index].reset_index(drop=True)

        # Separate positive and negative sample indices
        self.positive_idxs = self.folds[self.folds.exist_labels == 1].index.values
        self.negative_idxs = self.folds[self.folds.exist_labels == 0].index.values

        # Calculate sample counts
        self.n_positive = self.positive_idxs.shape[0]
        self.n_negative = int(self.n_positive * (1 - self.positive_proba) / self.positive_proba)
        
    def __iter__(self):
        """Generate sampling indices"""
        negative_sample = np.random.choice(self.negative_idxs, size=self.n_negative)
        shuffled = np.random.permutation(np.hstack((negative_sample, self.positive_idxs)))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.n_positive + self.n_negative
