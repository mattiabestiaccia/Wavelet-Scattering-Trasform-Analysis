import os
import random
from collections import defaultdict
from PIL import Image
from wavelet_lib.base import BaseWaveletDataset, WaveletProcessor

class TileDataset(BaseWaveletDataset):
    def __init__(self, dataset_root, processor: WaveletProcessor = None, transform=None, balance=True, allowed_extensions={'.jpg'}):
        super().__init__(
            root=dataset_root,
            processor=processor,
            transform=transform,
            balance=balance,
            allowed_extensions=allowed_extensions
        )
        # Initialize after parent class initialization
        self.data = self.load_data()
        self.wavelet_representations = self.prepare_wavelet_representation()

    def prepare_wavelet_representation(self):
        """Prepare wavelet representations for the dataset"""
        if not self.processor:
            return {}
        
        print("\nPreparing wavelet representations...")
        representations = {}
        total_files = len(self.samples)
        
        for idx, (filepath, _) in enumerate(self.samples, 1):
            if idx % 100 == 0:  # Stampa progresso ogni 100 file
                print(f"Processing file {idx}/{total_files} ({(idx/total_files)*100:.1f}%)")
            
            image = Image.open(filepath).convert('RGB')
            if self.transform:
                image = self.transform(image)
            coeffs = self.processor.compute_coefficients(image)
            representations[filepath] = coeffs
        
        print("Wavelet representations completed!")
        return representations
    
    def load_data(self):
        """Load raw data for wavelet processing"""
        print("\nLoading raw data...")
        data = {}
        total_files = len(self.samples)
        
        for idx, (filepath, _) in enumerate(self.samples, 1):
            if idx % 100 == 0:  # Stampa progresso ogni 100 file
                print(f"Loading file {idx}/{total_files} ({(idx/total_files)*100:.1f}%)")
            
            image = Image.open(filepath).convert('RGB')
            if self.transform:
                image = self.transform(image)
            data[filepath] = image
        
        print("Raw data loading completed!")
        return data

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

class DroneWaveletDataset(BaseWaveletDataset):
    def __init__(self, 
                 data_dir, 
                 processor: WaveletProcessor,
                 transform=None):
        self.data_dir = data_dir
        self.processor = processor
        self.transform = transform
        self.data = self.load_data()
        self.wavelet_representations = self.prepare_wavelet_representation()
    
    def prepare_wavelet_representation(self):
        """Common wavelet representation preparation"""
        representations = {}
        for img_id, img_data in self.data.items():
            coeffs = self.processor.compute_coefficients(img_data)
            representations[img_id] = coeffs
        return representations
    
    def load_data(self):
        """Basic data loading"""
        # Implementa caricamento base
        pass

class WaterSegmentationDataset(DroneWaveletDataset):
    """Dataset specifico per segmentazione dell'acqua"""
    def load_data(self):
        data = super().load_data()
        # Aggiungi caricamento maschere acqua
        return data

class VegetationDataset(DroneWaveletDataset):
    """Dataset specifico per analisi vegetazione"""
    def load_data(self):
        data = super().load_data()
        # Aggiungi caricamento dati vegetazione
        return data
