from wavelet_lib.base import BaseWaveletDataset, BaseWaveletModel


class WaveletTrainer:
    def __init__(self, 
                 model: BaseWaveletModel,
                 dataset: BaseWaveletDataset,
                 config: dict):
        self.model = model
        self.dataset = dataset
        self.config = config
        
    def train(self):
        """Training generico"""
        pass
    
    def evaluate(self):
        """Valutazione generica"""
        pass

class SegmentationTrainer(WaveletTrainer):
    """Trainer specifico per segmentazione"""
    def train(self):
        super().train()
        # Aggiungi logica specifica per training segmentazione

class ClassificationTrainer(WaveletTrainer):
    """Trainer specifico per classificazione"""
    def train(self):
        super().train()
        # Aggiungi logica specifica per training classificazione