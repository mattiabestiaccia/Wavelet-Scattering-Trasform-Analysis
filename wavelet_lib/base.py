from abc import ABC, abstractmethod
from kymatio.torch import Scattering2D
import torch.nn as nn
import os
import random
from collections import defaultdict
from PIL import Image
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from kymatio.torch import Scattering2D

# Constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScatteringPreprocessor(nn.Module):
    """
    A preprocessor module that applies the Scattering Wavelet Transform to input data.
    
    This module uses the Scattering2D transform from the Kymatio library to compute
    wavelet scattering coefficients. 

    Args:
        J (int): Number of scales for the scattering transform. Default is 2.
        shape (tuple): Shape of the input images. Default is (32, 32).
        max_order (int): Maximum order of scattering coefficients. Default is SCATTERING_ORDER.
        
    """
    def __init__(self, J=2, shape=(32, 32), max_order=2, **kwargs):
        super().__init__()
        self.scattering = Scattering2D(J=J, shape=shape, max_order=max_order).to(device)
        self.J = J
        self.shape = shape
        self.max_order = max_order
        # Calcola output_size usando un input dummy
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *shape).to(device)
            dummy_output = self.scattering(dummy_input)
            _, self.output_size, _, _ = dummy_output.shape[1:]
        
    def forward(self, x):
        S = self.scattering(x)
        batch_size, channels, coeffs, h, w = S.shape
        return S.view(batch_size, channels * coeffs, h, w)

class WaveletProcessor(ABC):
    """Base interface for wavelet processing"""
    @abstractmethod
    def compute_coefficients(self, input_data):
        pass
    
    @abstractmethod
    def preprocess(self, data):
        pass

class BaseWaveletDataset(torch.utils.data.Dataset, ABC):
    """Base interface for wavelet datasets with optional balancing capabilities"""
    def __init__(self, 
                 root, 
                 processor: WaveletProcessor = None,
                 transform=None, 
                 balance=True,
                 allowed_extensions={'.jpg'}):
        self.root = root
        self.processor = processor
        self.transform = transform
        self.allowed_extensions = allowed_extensions
        self.samples = []
        
        # Initialize class information
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Load data and prepare wavelet representations
        self.data = self.load_data()
        self.wavelet_representations = self.prepare_wavelet_representation()
        
        # Optional balancing
        if balance:
            self._balance_dataset()
        else:
            self._load_unbalanced_dataset()
            
    def _load_unbalanced_dataset(self):
        """Load dataset without balancing"""
        for cls in self.classes:
            cls_dir = os.path.join(self.root, cls)
            for fname in os.listdir(cls_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.allowed_extensions:
                    filepath = os.path.join(cls_dir, fname)
                    self.samples.append((filepath, self.class_to_idx[cls]))
    
    def _balance_dataset(self):
        """Balance the dataset by undersampling"""
        class_images = defaultdict(list)
        
        # Collect images per class
        for cls in self.classes:
            cls_dir = os.path.join(self.root, cls)
            for fname in os.listdir(cls_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.allowed_extensions:
                    filepath = os.path.join(cls_dir, fname)
                    class_images[cls].append((filepath, self.class_to_idx[cls]))
        
        # Balance by undersampling
        min_samples = min(len(images) for images in class_images.values())
        for cls, images in class_images.items():
            if len(images) > min_samples:
                selected_images = random.sample(images, min_samples)
                self.samples.extend(selected_images)
            else:
                self.samples.extend(images)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        filepath, label = self.samples[index]
        image = Image.open(filepath).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Apply wavelet processing if available
        if self.processor and filepath in self.wavelet_representations:
            image = self.wavelet_representations[filepath]
        elif self.processor:  # Compute and cache if processor exists but representation doesn't
            coeffs = self.processor.compute_coefficients(image)
            self.wavelet_representations[filepath] = coeffs
            image = coeffs
            
        return image, label
    
    def get_class_names(self):
        """Return list of class names"""
        return self.classes
    
    def get_class_distribution(self):
        """Return dictionary with number of samples per class"""
        distribution = defaultdict(int)
        for _, label in self.samples:
            class_name = self.classes[label]
            distribution[class_name] += 1
        return dict(distribution)
    
    @abstractmethod
    def prepare_wavelet_representation(self):
        """Prepare wavelet representations for the dataset"""
        pass
    
    @abstractmethod
    def load_data(self):
        """Load raw data for wavelet processing"""
        pass

class BaseWaveletModel(nn.Module):
    """Base class for all wavelet-based models"""
    def __init__(self, scattering_params):
        super().__init__()
        self.scattering = ScatteringPreprocessor(**scattering_params)
    
    @abstractmethod
    def forward(self, x):
        pass
