import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch

class WSTDatasetAnalyzer:
    """Analizzatore per dataset WST (Wavelet Scattering Transform)"""
    
    def __init__(self, pickle_path: str, save_dir: Optional[str] = None):
        """
        Inizializza l'analizzatore WST.
        
        Args:
            pickle_path: Percorso al file pickle del dataset
            save_dir: Directory dove salvare i risultati dell'analisi
        """
        self.pickle_path = pickle_path
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        self.dataset = self._load_dataset()
        self.analysis_results = None
    
    def _load_dataset(self) -> Dict[str, Any]:
        """Carica il dataset dal file pickle"""
        print(f"Caricamento dataset da: {self.pickle_path}")
        with open(self.pickle_path, 'rb') as f:
            return pickle.load(f)
    
    def analyze(self) -> Dict[str, Any]:
        """
        Esegue l'analisi completa del dataset.
        
        Returns:
            Dict contenente i risultati dell'analisi
        """
        print("\n" + "="*50)
        print("ANALISI DATASET WST")
        print("="*50)
        
        # Analisi delle informazioni generali
        general_info = self._analyze_general_info()
        
        # Analisi della distribuzione delle classi
        class_distribution = self._analyze_class_distribution()
        
        # Analisi delle rappresentazioni wavelet
        wavelet_stats = self._analyze_wavelet_representations()
        
        # Combina tutti i risultati
        self.analysis_results = {
            'general_info': general_info,
            'class_distribution': class_distribution,
            'wavelet_stats': wavelet_stats,
            'classes': self.dataset['classes']
        }
        
        return self.analysis_results
    
    def _analyze_general_info(self) -> Dict[str, Any]:
        """Analizza le informazioni generali del dataset"""
        print("\n1. INFORMAZIONI GENERALI:")
        info = {
            'total_samples': len(self.dataset['samples']),
            'num_classes': len(self.dataset['classes']),
            'available_classes': self.dataset['classes']
        }
        
        print(f"• Numero totale campioni: {info['total_samples']}")
        print(f"• Numero di classi: {info['num_classes']}")
        print(f"• Classi disponibili: {info['available_classes']}")
        
        return info
    
    def _analyze_class_distribution(self) -> Counter:
        """Analizza la distribuzione delle classi"""
        print("\n2. DISTRIBUZIONE DELLE CLASSI:")
        distribution = Counter([label for _, label in self.dataset['samples']])
        
        for class_name in self.dataset['classes']:
            class_idx = self.dataset['class_to_idx'][class_name]
            count = distribution[class_idx]
            percentage = (count / len(self.dataset['samples'])) * 100
            print(f"• {class_name}: {count} campioni ({percentage:.1f}%)")
        
        return distribution
    
    def _analyze_wavelet_representations(self) -> Dict[str, Any]:
        """Analizza le statistiche delle rappresentazioni wavelet"""
        print("\n3. ANALISI RAPPRESENTAZIONI WAVELET:")
        sample_path = list(self.dataset['wavelet_representations'].keys())[0]
        sample_repr = self.dataset['wavelet_representations'][sample_path]
        
        stats = {
            'shape': tuple(sample_repr.shape),
            'dtype': str(sample_repr.dtype),
            'min_value': float(sample_repr.min()),
            'max_value': float(sample_repr.max()),
            'mean': float(sample_repr.mean()),
            'std': float(sample_repr.std())
        }
        
        print(f"• Shape rappresentazione: {stats['shape']}")
        print(f"• Tipo dati: {stats['dtype']}")
        print(f"• Range valori: [{stats['min_value']:.3f}, {stats['max_value']:.3f}]")
        print(f"• Media: {stats['mean']:.3f}")
        print(f"• Deviazione standard: {stats['std']:.3f}")
        
        return stats
    
    def plot_analysis(self) -> None:
        """Crea e salva i grafici delle analisi"""
        if not self.analysis_results:
            raise ValueError("Esegui prima il metodo analyze()")
        
        self._plot_class_distribution()
        self._plot_wavelet_heatmaps()
    
    def _plot_class_distribution(self) -> None:
        """Crea e salva il grafico della distribuzione delle classi"""
        plt.figure(figsize=(12, 6))
        classes = self.dataset['classes']
        counts = [self.analysis_results['class_distribution'][i] for i in range(len(classes))]
        
        sns.barplot(x=classes, y=counts)
        plt.title('Distribuzione delle Classi nel Dataset')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Numero di Campioni')
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, 'class_distribution.png'))
        plt.close()
    
    def _plot_wavelet_heatmaps(self) -> None:
        """Crea e salva le heatmap delle rappresentazioni wavelet"""
        sample_path = list(self.dataset['wavelet_representations'].keys())[0]
        sample_repr = self.dataset['wavelet_representations'][sample_path]
        
        for channel in range(sample_repr.shape[0]):
            plt.figure(figsize=(10, 8))
            channel_data = sample_repr[channel, 0].cpu().numpy()
            sns.heatmap(channel_data, cmap='viridis')
            plt.title(f'Heatmap Rappresentazione Wavelet - Canale {channel}')
            plt.tight_layout()
            
            if self.save_dir:
                plt.savefig(os.path.join(self.save_dir, f'wavelet_heatmap_channel_{channel}.png'))
            plt.close()

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Restituisce le statistiche principali del dataset per l'uso in altri moduli.
        
        Returns:
            Dict contenente le statistiche principali del dataset
        """
        if not self.analysis_results:
            self.analyze()
            
        return {
            'mean': self.analysis_results['wavelet_stats']['mean'],
            'std': self.analysis_results['wavelet_stats']['std'],
            'shape': self.analysis_results['wavelet_stats']['shape'],
            'num_classes': self.analysis_results['general_info']['num_classes'],
            'classes': self.analysis_results['general_info']['available_classes']
        }