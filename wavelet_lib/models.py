import torch
import torch.nn as nn
from wavelet_lib.base import BaseWaveletModel
from torch import nn
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)


# class WaveletSegmenter(BaseWaveletModel):
#     """Modello per segmentazione basato su wavelet"""
#     def __init__(self, scattering_params, num_classes=1):
#         super().__init__(scattering_params)
#         self.decoder = self._build_decoder()
    
#     def forward(self, x):
#         coeffs = self.scattering(x)
#         return self.decoder(coeffs)

# class WaveletClassifier(BaseWaveletModel):
#     """Modello per classificazione basato su wavelet"""
#     def __init__(self, scattering_params, num_classes):
#         super().__init__(scattering_params)
#         self.classifier = self._build_classifier(num_classes)
    
#     def forward(self, x):
#         coeffs = self.scattering(x)
#         return self.classifier(coeffs)
    
class TileWaveletClassifier(BaseWaveletModel):
    def __init__(self, scattering_params, num_classes=7, in_channels=3):
        super().__init__(scattering_params)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.build()
        
    def build(self):
        # Assumiamo che i dati siano già nella forma [batch, channels, H, W]
        # dove channels = in_channels (es. 3*81 = 243 per RGB con 81 coefficienti)
        current_in_channels = self.in_channels
        
        cfg = [128, 128, 'M', 256, 256, 'M', 512, 512]
        layers = []
        
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(current_in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True)
                ]
                current_in_channels = v
                
        layers += [nn.AdaptiveAvgPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512 * 4, self.num_classes)

    def forward(self, x):
        # x è di dimensione [batch, channels, H, W]
        batch_size = x.size(0)
        
        x = self.features(x)
        x = x.view(batch_size, -1)
        return self.classifier(x)
