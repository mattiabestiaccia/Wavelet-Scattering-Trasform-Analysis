import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from pathlib import Path

def load_wst_dataset(pickle_path):
    """Carica il dataset dal file pickle"""
    print(f"Caricamento dataset da: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def analyze_dataset(dataset):
    """Analizza e mostra le informazioni principali del dataset"""
    print("\n" + "="*50)
    print("ANALISI DATASET WST")
    print("="*50)

    # 1. Informazioni Generali
    print("\n1. INFORMAZIONI GENERALI:")
    print(f"• Numero totale campioni: {len(dataset['samples'])}")
    print(f"• Numero di classi: {len(dataset['classes'])}")
    print(f"• Classi disponibili: {dataset['classes']}")

    # 2. Distribuzione delle Classi
    class_distribution = Counter([label for _, label in dataset['samples']])
    print("\n2. DISTRIBUZIONE DELLE CLASSI:")
    for class_name in dataset['classes']:
        class_idx = dataset['class_to_idx'][class_name]
        count = class_distribution[class_idx]
        percentage = (count / len(dataset['samples'])) * 100
        print(f"• {class_name}: {count} campioni ({percentage:.1f}%)")

    # 3. Analisi Rappresentazioni Wavelet
    print("\n3. ANALISI RAPPRESENTAZIONI WAVELET:")
    sample_path = list(dataset['wavelet_representations'].keys())[0]
    sample_repr = dataset['wavelet_representations'][sample_path]
    print(f"• Shape rappresentazione: {sample_repr.shape}")
    print(f"• Tipo dati: {sample_repr.dtype}")
    print(f"• Range valori: [{sample_repr.min():.3f}, {sample_repr.max():.3f}]")
    print(f"• Media: {sample_repr.mean():.3f}")
    print(f"• Deviazione standard: {sample_repr.std():.3f}")

    return {
        'class_distribution': class_distribution,
        'sample_repr': sample_repr,
        'classes': dataset['classes']
    }

def plot_dataset_info(analysis_results, save_dir=None):
    """Crea e salva i grafici delle analisi"""
    # 1. Distribuzione delle Classi
    plt.figure(figsize=(12, 6))
    classes = analysis_results['classes']
    counts = [analysis_results['class_distribution'][i] for i in range(len(classes))]
    
    sns.barplot(x=classes, y=counts)
    plt.title('Distribuzione delle Classi nel Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Numero di Campioni')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'class_distribution.png'))
    plt.close()

    # 2. Heatmap della Prima Rappresentazione Wavelet
    sample_repr = analysis_results['sample_repr']
    
    # Plot separate heatmaps for each channel
    for channel in range(3):
        plt.figure(figsize=(10, 8))
        # Take the first slice of the channel dimension and convert to numpy
        channel_data = sample_repr[channel, 0].cpu().numpy()
        sns.heatmap(channel_data, cmap='viridis')
        plt.title(f'Heatmap Rappresentazione Wavelet - Canale {channel}')
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'wavelet_heatmap_channel_{channel}.png'))
        plt.close()

def main():
    # Configurazione
    pickle_path = "/home/brus/Projects/wavelet/datasets/processed_datasets/processed_tile_dataset.pkl"
    save_dir = "dataset_analysis"

    # Crea directory per i risultati
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Carica e analizza il dataset
    dataset = load_wst_dataset(pickle_path)
    analysis_results = analyze_dataset(dataset)

    # Crea e salva i grafici
    plot_dataset_info(analysis_results, save_dir)
    
    print(f"\nAnalisi completata. Risultati salvati in: {save_dir}")

if __name__ == "__main__":
    main()
