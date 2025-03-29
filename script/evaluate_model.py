import torch
import os
import argparse
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import tqdm

from wavelet_lib.models import TileWaveletClassifier
from wavelet_lib.transforms import process_image_for_classification
from script.classify import Config, LoadedTileDataset

class ModelEvaluator:
    def __init__(self, model_path: str, dataset_path: str):
        """
        Inizializza l'evaluator del modello.
        
        Args:
            model_path: Percorso al modello salvato (.pth)
            dataset_path: Percorso al dataset pickle per ottenere le classi
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Carica il dataset solo per ottenere le classi e le statistiche
        self.dataset = LoadedTileDataset(dataset_path)
        self.mean = self.dataset.mean
        self.std = self.dataset.std
        self.classes = self.dataset.classes
        
        # Carica il modello
        scattering_params = {
            "J": 2,
            "shape": (32, 32),
            "max_order": 2
        }
        self.model = TileWaveletClassifier(
            scattering_params=scattering_params,
            num_classes=Config.NUM_CLASSES
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        print(f"Modello caricato da: {model_path}")
        print(f"Classi disponibili: {self.classes}")

    def predict_image(self, image_path: str, show_results: bool = True) -> Tuple[str, float]:
        """
        Classifica una singola immagine.
        
        Args:
            image_path: Percorso all'immagine da classificare
            show_results: Se True, mostra l'immagine con la predizione
            
        Returns:
            Tupla (classe predetta, confidenza)
        """
        # Processa l'immagine
        image_tensor = process_image_for_classification(image_path)
        
        # Normalizza usando media e deviazione standard del training
        image_tensor = (image_tensor - self.mean) / self.std
        
        # Aggiungi dimensione batch
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Predizione
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
            
        predicted_class = self.classes[predicted.item()]
        confidence = confidence.item()
        
        # Visualizza risultati
        if show_results:
            self._display_prediction(image_path, predicted_class, confidence)
            
        return predicted_class, confidence

    def evaluate_directory(self, dir_path: str, save_results: bool = True) -> Dict[str, List[Tuple[str, float]]]:
        """
        Valuta tutte le immagini in una directory.
        
        Args:
            dir_path: Percorso alla directory contenente le immagini
            save_results: Se True, salva i risultati in un file
            
        Returns:
            Dizionario con i risultati per ogni immagine
        """
        results = {}
        image_files = [f for f in Path(dir_path).glob("*") 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']]
        
        print(f"\nValutazione di {len(image_files)} immagini in {dir_path}")
        
        for img_path in tqdm.tqdm(image_files):
            try:
                predicted_class, confidence = self.predict_image(str(img_path), show_results=False)
                results[img_path.name] = (predicted_class, confidence)
            except Exception as e:
                print(f"\nErrore nell'elaborazione di {img_path.name}: {str(e)}")
                results[img_path.name] = ("error", 0.0)
        
        # Salva risultati
        if save_results:
            self._save_results(results, dir_path)
            
        return results

    def _display_prediction(self, image_path: str, predicted_class: str, confidence: float):
        """Visualizza l'immagine con la predizione"""
        img = Image.open(image_path)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f'Predizione: {predicted_class}\nConfidenza: {confidence:.2%}')
        plt.axis('off')
        plt.show()

    def _save_results(self, results: Dict[str, Tuple[str, float]], dir_path: str):
        """Salva i risultati in un file di testo"""
        output_path = os.path.join(dir_path, "classification_results.txt")
        
        with open(output_path, 'w') as f:
            f.write("RISULTATI CLASSIFICAZIONE\n")
            f.write("=" * 50 + "\n\n")
            
            for img_name, (pred_class, conf) in results.items():
                f.write(f"Immagine: {img_name}\n")
                f.write(f"Classe: {pred_class}\n")
                f.write(f"Confidenza: {conf:.2%}\n")
                f.write("-" * 30 + "\n")
        
        print(f"\nRisultati salvati in: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Valuta il modello su nuove immagini')
    parser.add_argument('--model', type=str, required=True,
                      help='Percorso al modello addestrato (.pth)')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Percorso al dataset pickle')
    parser.add_argument('--image', type=str,
                      help='Percorso a una singola immagine da classificare')
    parser.add_argument('--dir', type=str,
                      help='Percorso a una directory di immagini da classificare')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.model, args.dataset)
    
    if args.image:
        predicted_class, confidence = evaluator.predict_image(args.image)
        print(f"\nRisultati per {args.image}:")
        print(f"Classe predetta: {predicted_class}")
        print(f"Confidenza: {confidence:.2%}")
    
    elif args.dir:
        results = evaluator.evaluate_directory(args.dir)
        print("\nValutazione completata!")
    
    else:
        parser.error("Specificare --image o --dir")

if __name__ == '__main__':
    main()
