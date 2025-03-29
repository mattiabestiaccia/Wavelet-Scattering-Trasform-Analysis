from kymatio.torch import Scattering2D
from wavelet_lib.base import WaveletProcessor

class DroneImageWaveletProcessor(WaveletProcessor):
    def __init__(self, J=2, shape=(32, 32)):
        self.J = J
        self.shape = shape
        self.scattering = Scattering2D(J=J, shape=shape)
        
    def compute_coefficients(self, input_data):
        """Compute wavelet scattering coefficients"""
        return self.scattering(input_data)
    
    def preprocess(self, data):
        """Common preprocessing for drone images"""
        # Implementa preprocessing comune
        pass

class MultispectralWaveletProcessor(DroneImageWaveletProcessor):
    """Specialized processor for multispectral images"""
    def preprocess(self, data):
        super().preprocess(data)
        # Aggiungi preprocessing specifico per immagini multispettrali