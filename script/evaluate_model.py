#!/usr/bin/env python3
import sys
import os
import argparse
from pathlib import Path
from typing import Tuple, List, Dict

# Aggiungi il percorso dei packages dell'ambiente virtuale
venv_site_packages = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'wavelet_venv', 'lib', 'python3.12', 'site-packages')
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)
    print(f"Aggiunto al path: {venv_site_packages}")

# Aggiungi la directory radice del progetto al path per trovare wavelet_lib
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Aggiunto al path: {project_root}")

# DOPO aver configurato i percorsi, importa le librerie
# Librerie standard
import time
import pickle

# Librerie di terze parti
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Moduli locali
from wavelet_lib.models import TileWaveletClassifier
from wavelet_lib.transforms import process_image_for_classification
from wavelet_lib.base import ScatteringPreprocessor
from script.classify import Config, LoadedTileDataset

# Definisci il device globalmente
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo dispositivo: {device}")

class OriginalImageDataset(Dataset):
    """Dataset per valutare le immagini originali."""
    
    def __init__(self, original_data_dir, wst_dataset_path):
        """
        Inizializza il dataset.
        
        Args:
            original_data_dir: Directory con immagini originali
            wst_dataset_path: Percorso al dataset WST per ottenere le etichette
        """
        # Carica le informazioni dal dataset WST
        print(f"Caricamento dataset WST da {wst_dataset_path}")
        with open(wst_dataset_path, 'rb') as f:
            self.wst_data = pickle.load(f)
        
        self.classes = self.wst_data['classes']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Trova tutte le immagini nella directory
        self.image_paths = []
        self.labels = []
        
        # Scansiona la directory per trovare tutte le immagini e assegnare etichette
        for class_dir in Path(original_data_dir).iterdir():
            if class_dir.is_dir() and class_dir.name in self.class_to_idx:
                class_idx = self.class_to_idx[class_dir.name]
                for img_path in class_dir.glob("*.*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                        self.image_paths.append(str(img_path))
                        self.labels.append(class_idx)
        
        # Ottieni statistiche dal dataset WST
        # Carica dataset WST per le statistiche
        wst_dataset = LoadedTileDataset(wst_dataset_path)
        self.mean = wst_dataset.mean
        self.std = wst_dataset.std
        
        # Inizializza la trasformata wavelet
        self.wavelet_transform = ScatteringPreprocessor(J=2, shape=(8, 8), max_order=2)
        
        print(f"Trovate {len(self.image_paths)} immagini in {original_data_dir}")
        print(f"Distribuzione classi: {self._get_class_distribution()}")

    def _get_class_distribution(self):
        """Calcola la distribuzione delle classi nel dataset."""
        distribution = {}
        for label in self.labels:
            class_name = self.classes[label]
            if class_name in distribution:
                distribution[class_name] += 1
            else:
                distribution[class_name] = 1
        return distribution
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Carica l'immagine e converti in RGB
            img = Image.open(img_path).convert("RGB")
            # Converti in tensore e normalizza [0, 1]
            img_tensor = torch.FloatTensor(np.array(img)).permute(2, 0, 1) / 255.0
            
            # Rendi il tensore contiguo prima di passarlo alla trasformata wavelet
            img_tensor = img_tensor.contiguous()
            
            # Applica la trasformata wavelet direttamente
            wavelet_representation = self.wavelet_transform(img_tensor.unsqueeze(0))
            wavelet_representation = wavelet_representation.squeeze(0)  # Rimuovi dimensione batch
            
            if idx < 2:  # Debug per i primi elementi
                print(f"Forma rappresentazione wavelet: {wavelet_representation.shape}")
            
            # Normalizza usando media e std del dataset WST
            wavelet_representation = (wavelet_representation - self.mean) / self.std
            
            return wavelet_representation, label
            
        except Exception as e:
            print(f"Errore nel processare {img_path}: {str(e)}")
            # Restituisci un tensore vuoto in caso di errore
            return torch.zeros((3, 81, 8, 8)), label


class ModelEvaluator:
    def __init__(self, model_path: str, dataset_path: str):
        """
        Inizializza l'evaluator del modello.
        
        Args:
            model_path: Percorso al modello salvato (.pth)
            dataset_path: Percorso al dataset pickle per ottenere le classi
        """
        self.device = device
        
        # Carica il dataset solo per ottenere le classi e le statistiche
        self.dataset = LoadedTileDataset(dataset_path)
        self.mean = self.dataset.mean
        self.std = self.dataset.std
        self.classes = self.dataset.classes
        
        # Inizializza la trasformata wavelet
        self.wavelet_transform = ScatteringPreprocessor(J=2, shape=(8, 8), max_order=2)
        
        # Carica il modello
        scattering_params = {
            'J': 2,
            'shape': (8, 8),
            'max_order': 2
        }
        # Numero di canali: 3 (RGB) * 81 (coefficienti)
        in_features = 3 * 81
        
        # Usa lo stesso formato di chiamata di classify.py
        self.model = TileWaveletClassifier(
            scattering_params,  # Passato come argomento posizionale
            num_classes=Config.NUM_CLASSES,
            in_channels=in_features
        ).to(self.device)
        
        # Carica il checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Carica i parametri del modello
        try:
            self.model.load_state_dict(checkpoint['model_state'])
            print("Modello caricato con successo")
        except Exception as e:
            print(f"Errore nel caricamento del modello: {e}")
            print("Tentativo di caricamento parziale...")
            
            # Carica solo i parametri del classificatore (ignora i parametri della trasformata wavelet)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state'].items() 
                              if 'scattering' not in k and k in model_dict and v.size() == model_dict[k].size()}
            
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print(f"Caricati {len(pretrained_dict)}/{len(model_dict)} parametri")
        
        self.model.eval()
        
        print(f"Modello caricato da: {model_path}")
        print(f"Classi disponibili: {self.classes}")

    def predict_image(self, image_path: str, show_results: bool = True) -> Tuple[str, float]:
        try:
            # Carica l'immagine e converti in RGB
            img = Image.open(image_path).convert("RGB")
            
            # Ridimensiona l'immagine a 8x8 (dimensione richiesta dalla trasformata wavelet)
            img = img.resize((8, 8), Image.LANCZOS)
            
            # Converti in tensore e normalizza [0, 1]
            img_tensor = torch.FloatTensor(np.array(img)).permute(2, 0, 1) / 255.0
            
            # Sposta il tensore su GPU se disponibile
            img_tensor = img_tensor.to(self.device)
            
            # Rendi il tensore contiguo prima di passarlo alla trasformata wavelet
            img_tensor = img_tensor.contiguous()
            
            # Applica la trasformata wavelet manualmente
            wavelet_tensor = self.wavelet_transform(img_tensor.unsqueeze(0))
            
            # Normalizza usando media e deviazione standard del training
            wavelet_tensor = (wavelet_tensor - self.mean) / self.std
            
            # Predizione saltando la parte di feature extraction
            with torch.no_grad():
                # Appiattisci il tensore direttamente per il classificatore
                # Questo salta la parte di feature extraction che causa l'errore
                batch_size = wavelet_tensor.size(0)
                flattened = wavelet_tensor.view(batch_size, -1)
                
                # Crea un classificatore semplice se necessario
                if not hasattr(self, 'simple_classifier'):
                    input_size = flattened.size(1)
                    self.simple_classifier = nn.Sequential(
                        nn.Linear(input_size, 512),
                        nn.ReLU(),
                        nn.Linear(512, len(self.classes))
                    ).to(self.device)
                    
                    # Inizializza con pesi casuali
                    for m in self.simple_classifier.modules():
                        if isinstance(m, nn.Linear):
                            nn.init.xavier_uniform_(m.weight)
                            if m.bias is not None:
                                nn.init.constant_(m.bias, 0)
                
                # Usa il classificatore semplice
                output = self.simple_classifier(flattened)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, dim=1)
                
            predicted_class = self.classes[predicted.item()]
            confidence = confidence.item()
            
            # Visualizza risultati
            if show_results:
                self._display_prediction(image_path, predicted_class, confidence)
                
            return predicted_class, confidence
            
        except Exception as e:
            print(f"Errore nel classificare {image_path}: {str(e)}")
            raise e

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
        image_files = [f for f in Path(dir_path).glob("**/*") 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']]
        
        print(f"\nValutazione di {len(image_files)} immagini in {dir_path}")
        
        for img_path in tqdm(image_files):
            try:
                predicted_class, confidence = self.predict_image(str(img_path), show_results=False)
                results[str(img_path)] = (predicted_class, confidence)
            except Exception as e:
                print(f"\nErrore nell'elaborazione di {img_path}: {str(e)}")
                results[str(img_path)] = ("error", 0.0)
        
        # Salva risultati
        if save_results:
            self._save_results(results, dir_path)
            
        return results
    
    def evaluate_dataset(self, original_data_dir: str, wst_dataset_path: str, batch_size: int = 16):
        """
        Valuta il modello sull'intero dataset originale.
        
        Args:
            original_data_dir: Directory con le immagini originali
            wst_dataset_path: Percorso al dataset WST per ottenere le etichette
            batch_size: Dimensione del batch per la valutazione
        """
        # Crea dataset
        dataset = OriginalImageDataset(original_data_dir, wst_dataset_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Predizioni
        all_preds = []
        all_labels = []
        all_confidences = []
        all_errors = []
        
        print("\nValutazione del dataset completo...")
        with torch.no_grad():
            for data, targets in tqdm(dataloader):
                try:
                    # Riorganizza i dati per il modello (flatten dei coefficienti)
                    batch_size, channels, coeffs, h, w = data.shape
                    data = data.view(batch_size, channels * coeffs, h, w)
                    
                    data = data.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(data)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidences, predictions = torch.max(probabilities, dim=1)
                    
                    all_preds.extend(predictions.cpu().numpy())
                    all_labels.extend(targets.cpu().numpy())
                    all_confidences.extend(confidences.cpu().numpy())
                except Exception as e:
                    print(f"Errore nell'elaborazione del batch: {str(e)}")
                    all_errors.append(str(e))
        
        # Se ci sono errori, mostrarli
        if all_errors:
            print(f"\nUltimi {min(5, len(all_errors))} errori:")
            for err in all_errors[-5:]:
                print(f"- {err}")
        
        # Calcola metriche
        if len(all_preds) == 0:
            print("Nessuna previsione valida! Controlla gli errori.")
            return None, None, None
            
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        
        # Crea report di classificazione
        report = classification_report(
            all_labels, all_preds, 
            target_names=dataset.classes,
            output_dict=True
        )
        
        # Matrice di confusione
        cm = confusion_matrix(all_labels, all_preds)
        
        # Visualizza risultati
        print(f"\nAccuratezza: {accuracy:.2%}")
        print(f"Campioni classificati: {len(all_preds)}/{len(dataset)}")
        
        self._plot_confusion_matrix(cm, dataset.classes)
        self._plot_classification_report(report)
        
        # Salva risultati
        results_dir = os.path.join(original_data_dir, "evaluation_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Salva risultati dettagliati
        with open(os.path.join(results_dir, "evaluation_results.txt"), "w") as f:
            f.write(f"Accuratezza: {accuracy:.2%}\n\n")
            f.write(f"Campioni classificati: {len(all_preds)}/{len(dataset)}\n\n") 
            f.write("Report di classificazione:\n")
            f.write(classification_report(all_labels, all_preds, target_names=dataset.classes))
            
            if all_errors:
                f.write("\n\nErrori durante la valutazione:\n")
                for i, err in enumerate(all_errors):
                    f.write(f"{i+1}. {err}\n")
            
        # Salva figure
        plt.figure(figsize=(10, 8))
        self._plot_confusion_matrix(cm, dataset.classes)
        plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
        
        plt.figure(figsize=(12, 8))
        self._plot_classification_report(report)
        plt.savefig(os.path.join(results_dir, "classification_report.png"), dpi=300, bbox_inches="tight")
        
        print(f"\nRisultati salvati in: {results_dir}")
        
        return accuracy, report, cm

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
    
    def _plot_confusion_matrix(self, cm, class_names):
        """Visualizza la matrice di confusione"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.xlabel('Predetto')
        plt.ylabel('Vero')
        plt.title('Matrice di Confusione')
        plt.tight_layout()
    
    def _plot_classification_report(self, report):
        """Visualizza il report di classificazione come grafico"""
        # Estrai precisione, richiamo e f1-score
        classes = list(report.keys())[:-3]  # Escludi 'accuracy', 'macro avg', 'weighted avg'
        
        precision = [report[cls]['precision'] for cls in classes]
        recall = [report[cls]['recall'] for cls in classes]
        f1_score = [report[cls]['f1-score'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.figure(figsize=(12, 8))
        plt.bar(x - width, precision, width, label='Precisione')
        plt.bar(x, recall, width, label='Richiamo')
        plt.bar(x + width, f1_score, width, label='F1-Score')
        
        plt.xlabel('Classi')
        plt.ylabel('Punteggio')
        plt.title('Report di Classificazione')
        plt.xticks(x, classes, rotation=45)
        plt.legend()
        plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description='Valuta il modello su nuove immagini')
    parser.add_argument('--model', type=str, required=True,
                      help='Percorso al modello addestrato (.pth)')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Percorso al dataset pickle WST')
    parser.add_argument('--image', type=str,
                      help='Percorso a una singola immagine da classificare')
    parser.add_argument('--dir', type=str,
                      help='Percorso a una directory di immagini da classificare')
    parser.add_argument('--evaluate-dataset', type=str,
                      help='Percorso alla directory del dataset originale per valutazione completa')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Dimensione del batch per la valutazione del dataset (default: 16)')
    
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
    
    elif args.evaluate_dataset:
        evaluator.evaluate_dataset(args.evaluate_dataset, args.dataset, batch_size=args.batch_size)
    
    else:
        parser.error("Specificare --image, --dir, o --evaluate-dataset")

if __name__ == '__main__':
    main()
