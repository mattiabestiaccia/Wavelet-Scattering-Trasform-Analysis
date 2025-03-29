#!/usr/bin/env python3
import sys
import os
import argparse

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

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import signal
from wavelet_lib.models import TileWaveletClassifier

# Configurazioni
class Config:
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 100
    PATIENCE = 10
    NUM_CLASSES = 7
    INPUT_SHAPE = (3, 81, 8, 8)
    SAVE_DIR = "model_output_4 classes"
    CHECKPOINT_FILE = os.path.join(SAVE_DIR, "checkpoint.pth")
    BEST_MODEL_FILE = os.path.join(SAVE_DIR, "best_model.pth")
    FINAL_MODEL_FILE = os.path.join(SAVE_DIR, "final_model.pth")

# Gestione interrupt
class TrainingInterrupted(Exception):
    pass

def handle_interrupt(signum, frame):
    print("\nRicevuto interrupt, salvo lo stato corrente...")
    raise TrainingInterrupted

# Carica o inizializza checkpoint
def load_checkpoint():
    if os.path.exists(Config.CHECKPOINT_FILE):
        print(f"Trovato checkpoint esistente: {Config.CHECKPOINT_FILE}")
        checkpoint = torch.load(Config.CHECKPOINT_FILE)
        print(f"Checkpoint caricato (epoca {checkpoint['epoch']})")
        
        # Verifica la presenza di tutte le chiavi necessarie
        required_keys = ['epoch', 'model_state', 'optimizer_state', 'metrics', 'rng_state']
        if not all(key in checkpoint for key in required_keys):
            print("Checkpoint corrotto o incompleto, inizierò un nuovo training")
            return None
            
        return checkpoint
    return None

def save_checkpoint(epoch, model, optimizer, metrics):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'metrics': metrics,
        'rng_state': torch.get_rng_state(),
    }
    
    # Aggiungi lo stato CUDA solo se disponibile
    if torch.cuda.is_available():
        checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
    else:
        checkpoint['cuda_rng_state'] = None
    
    torch.save(checkpoint, Config.CHECKPOINT_FILE)

# Carica il dataset
class LoadedTileDataset(Dataset):
    def __init__(self, pickle_path):
        print(f"Caricamento dataset da {pickle_path}")
        with open(pickle_path, 'rb') as f:
            saved_data = pickle.load(f)
            
        self.wavelet_representations = saved_data['wavelet_representations']
        self.samples = saved_data['samples']
        self.classes = saved_data['classes']
        
        # Calcola statistiche reali
        all_data = torch.stack([self.wavelet_representations[fp] for fp, _ in self.samples])
        self.mean = all_data.mean()
        self.std = all_data.std()
        
        print("\nInformazioni Dataset:")
        print(f"Numero campioni: {len(self.samples)}")
        print(f"Classi: {self.classes}")
        print(f"Media: {self.mean:.4f}, Dev Std: {self.std:.4f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        image = self.wavelet_representations[filepath]
        return (image - self.mean) / self.std, label

# Funzioni di training
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for data, target in tqdm(loader, desc='Training'):
        # Stampa la forma dei dati
        print(f"Forma dei dati originale: {data.shape}")
        
        # Riorganizza i dati: da [batch, channels, coeffs, h, w] a [batch, channels*coeffs, h, w]
        batch_size, channels, coeffs, h, w = data.shape
        data = data.view(batch_size, channels * coeffs, h, w)
        
        print(f"Forma dei dati riorganizzata: {data.shape}")
        
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * target.size(0)
        correct += output.argmax(dim=1).eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for data, target in loader:
            # Riorganizza i dati: da [batch, channels, coeffs, h, w] a [batch, channels*coeffs, h, w]
            batch_size, channels, coeffs, h, w = data.shape
            data = data.view(batch_size, channels * coeffs, h, w)
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() * target.size(0)
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / total, correct / total

# Funzioni per i grafici
def plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies, block=False):
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_title('Loss over epochs')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_title('Accuracy over epochs')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show(block=block)
    if not block:
        plt.pause(0.1)

def save_plots(train_losses, train_accuracies, test_losses, test_accuracies, save_dir):
    plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies, block=True)
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Main
def main(resume=False):
    # Setup iniziale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    # Percorso al dataset processato
    dataset_path = "/home/brus/Projects/wavelet/datasets/HPL_images/custom_datasets_WST/processed_tile_dataset.pkl"
    
    # Se il dataset non esiste, mostra un messaggio e termina
    if not os.path.exists(dataset_path):
        print(f"ERRORE: Il dataset {dataset_path} non esiste.")
        print("Prima di eseguire questo script, è necessario creare il dataset processato.")
        return
    
    # Registra handler per interrupt
    signal.signal(signal.SIGINT, handle_interrupt)
    
    # Carica dataset (sempre da capo per consistenza)
    dataset = LoadedTileDataset(dataset_path)
    
    # Split dataset (deve essere sempre lo stesso)
    train_idx, test_idx = train_test_split(
        range(len(dataset)), 
        test_size=0.2, 
        random_state=42, 
        stratify=[label for _, label in dataset.samples]
    )
    
    train_loader = DataLoader(Subset(dataset, train_idx), 
                            batch_size=Config.BATCH_SIZE, 
                            shuffle=True, 
                            num_workers=4,
                            pin_memory=True)
    
    test_loader = DataLoader(Subset(dataset, test_idx),
                           batch_size=Config.BATCH_SIZE,
                           shuffle=False,
                           num_workers=4,
                           pin_memory=True)

    # Inizializza modello e ottimizzatore
    scattering_params = {
        'J': 2,
        'shape': (8, 8),
        'max_order': 2
    }
    # Calcola il numero di canali corretto: 3 (RGB) * 81 (coefficienti)
    in_features = 3 * 81
    
    # Modifico l'istanziazione del modello
    model = TileWaveletClassifier(scattering_params, num_classes=Config.NUM_CLASSES, in_channels=in_features).to(device)
    optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    # Carica checkpoint se richiesto
    start_epoch = 0
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # Modifica questa parte nel main():
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            metrics = checkpoint['metrics']
            torch.set_rng_state(checkpoint['rng_state'])
            
            # CORREZIONE: Controlla se c'è uno stato CUDA prima di usarlo
            if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint and checkpoint['cuda_rng_state'] is not None:
                torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
            
            print(f"Ripristinato training da epoca {start_epoch}")
        else:
            print("Nessun checkpoint trovato, inizio nuovo training")
    # Training loop
    best_loss = min(metrics['test_loss']) if metrics['test_loss'] else float('inf')
    patience_counter = 0

    print("Inizio training...")
    start_time = time.time()

    try:
        for epoch in range(start_epoch, Config.EPOCHS):
            # Training
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            # Aggiorna metriche
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            metrics['test_loss'].append(test_loss)
            metrics['test_acc'].append(test_acc)
            
            # Aggiorna scheduler
            scheduler.step(test_loss)
            
            # Stampa progresso
            print(f"\nEpoch {epoch+1}/{Config.EPOCHS}:")
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
            
            # Salva checkpoint ogni epoca
            save_checkpoint(epoch, model, optimizer, metrics)
            
            # Salva miglior modello
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'metrics': metrics,
                }, Config.BEST_MODEL_FILE)
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"\nEarly stopping dopo {epoch+1} epoche")
                    break
            
            # Visualizza grafici
            if (epoch + 1) % 5 == 0 or epoch == Config.EPOCHS - 1:
                plot_metrics(metrics['train_loss'], metrics['train_acc'],
                            metrics['test_loss'], metrics['test_acc'],
                            block=False)

    except TrainingInterrupted:
        print("\nTraining interrotto, stato salvato in checkpoint")
        # Mostra le metriche fino all'epoca corrente
        plot_metrics(metrics['train_loss'], metrics['train_acc'],
                    metrics['test_loss'], metrics['test_acc'],
                    block=True)
        return

    # Salva output finali
    print("\nSalvataggio output finali...")
    torch.save({
        'model_state': model.state_dict(),
        'metrics': metrics,
    }, Config.FINAL_MODEL_FILE)
    
    save_plots(metrics['train_loss'], metrics['train_acc'],
              metrics['test_loss'], metrics['test_acc'],
              Config.SAVE_DIR)
    
    # Rimuovi checkpoint al completamento
    if os.path.exists(Config.CHECKPOINT_FILE):
        os.remove(Config.CHECKPOINT_FILE)
    
    print(f"\nTraining completato in {(time.time()-start_time)/60:.2f} minuti")
    print(f"Risultati salvati in: {os.path.abspath(Config.SAVE_DIR)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Ripristina training da checkpoint')
    args = parser.parse_args()
    
    main(resume=args.resume)
