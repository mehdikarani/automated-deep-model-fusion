import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import shutil
import logging
from typing import List, Tuple, Dict

import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.cuda.amp import autocast, GradScaler  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import torch
import torch.nn as nn
from typing import List, Dict

class ANNModel:
    class CustomANN(nn.Module):
        def __init__(self, input_size: int, output_size: int, hidden_layers: List[int],
                     residual_connections: Dict[int, Dict] = {},
                     dropout_params: Dict[int, float] = {},
                     hidden_activation: str = 'ReLU',
                     seed: int = None, scale=False):
            super(ANNModel.CustomANN, self).__init__()
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

            self.activation_functions = {
                'ReLU': nn.ReLU(inplace=True),
                'ELU': nn.ELU(inplace=True),
                'GELU': nn.GELU(),  # GELU doesn't have an inplace option
                'SiLU': nn.SiLU(inplace=True),
                'LeakyReLU': nn.LeakyReLU(inplace=True)
            }
            self.hidden_activation = self.activation_functions[hidden_activation]

            self.input_layer = nn.Linear(input_size, hidden_layers[0])
            self.hidden_layers = nn.ModuleList()
            self.batch_norm_layers = nn.ModuleDict()
            self.dropout_layers = nn.ModuleDict()
            self.skip_connections = nn.ModuleDict()

            in_features = hidden_layers[0]
            for i, h in enumerate(hidden_layers[1:], 1):  # Start enumeration from 1
                self.hidden_layers.append(nn.Linear(in_features, h))
                if i in residual_connections:
                    self.batch_norm_layers[str(i)] = nn.BatchNorm1d(h)  # Removed inplace=True
                    for source_layer in residual_connections[i]['sources']:
                        source_features = input_size if source_layer == -1 else hidden_layers[source_layer]
                        self.skip_connections[f"{source_layer}->{i}"] = nn.Linear(source_features, h)
                if i in dropout_params:
                    self.dropout_layers[str(i)] = nn.Dropout(dropout_params[i])
                in_features = h
            
            
            if scale:
                self.output_layer = nn.Sequential(
                    nn.Linear(in_features, output_size),
                    nn.Sigmoid(),
                    nn.Linear(output_size, output_size)
                )
            else:
                self.output_layer = nn.Sequential(
                    nn.Linear(in_features, output_size),
                    nn.Linear(output_size, output_size)
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            activations = [x]
            x = self.hidden_activation(self.input_layer(x))
            activations.append(x)

            for i, hidden_layer in enumerate(self.hidden_layers, 1):
                layer_input = hidden_layer(x)
                
                # Apply residual connections if any
                if any(f"{src}->{i}" in self.skip_connections for src in range(-1, i)):
                    residual = sum(self.skip_connections[f"{src}->{i}"](activations[src+1])
                                   for src in range(-1, i)
                                   if f"{src}->{i}" in self.skip_connections)
                    layer_input = layer_input + residual

                    # Apply batch normalization only for layers with residual connections
                    if str(i) in self.batch_norm_layers:
                        layer_input = self.batch_norm_layers[str(i)](layer_input)
                
                # Apply activation
                x = self.hidden_activation(layer_input)
                
                # Apply dropout if specified for this layer
                if str(i) in self.dropout_layers:
                    x = self.dropout_layers[str(i)](x)
                
                activations.append(x)

            return self.output_layer(x)

    def create_model(self, input_size: int, output_size: int, hidden_layers: List[int],
                     residual_connections: Dict[int, Dict] = {},
                     dropout_params: Dict[int, float] = {},
                     hidden_activation: str = 'ReLU',
                     seed: int = None, scale=True):
        return self.CustomANN(input_size, output_size, hidden_layers,
                              residual_connections, dropout_params, hidden_activation, seed,scale=scale)


    def train_model(self,
                    train_loader, val_loader,
                    epochs: int,
                    learning_rate: float, optimizer_name: str,
                    optimizer_params: Dict = None,
                    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                    criterion = None, patience: int = 5,
                    model: nn.Module = None,
                    l2_lambda: float = 0.0,
                    epoch_interval: int = 50) -> Tuple[nn.Module, List[float], List[float]]:
        
        if optimizer_params is None:
            optimizer_params = {}
        
        if model is None:
            raise ValueError("No model provided for training.")
        
        model.to(device)
        
        
        optimizer = self._create_optimizer(optimizer_name, model.parameters(), learning_rate, optimizer_params, l2_lambda)
        
        # Add ReduceLROnPlateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        model.train()
        loss_hist_train = []
        loss_hist_valid = []
        best_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            
            # Update batch size if necessary
            # if epoch % epoch_interval == 0 and epoch > 0:
            #     batch_size *= 2
            #     # print(batch_size)
            #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            epoch_avg_train_loss=epoch_loss / len(train_loader)
            loss_hist_train.append(epoch_avg_train_loss)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            epoch_avg_val_loss = val_loss / len(val_loader)
            loss_hist_valid.append(epoch_avg_val_loss)
            
            # Step the scheduler
            scheduler.step(epoch_avg_train_loss)
            
            if epoch_avg_val_loss < best_loss:
                best_loss = epoch_avg_val_loss
                epochs_no_improve = 0
                best_model = model.state_dict()
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        model.load_state_dict(best_model)
        return model, loss_hist_train, loss_hist_valid

    def _create_dataloaders(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
                            batch_size: int, device: str) -> Tuple[DataLoader, DataLoader]:
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device), 
                                      torch.tensor(y_train, dtype=torch.float32).to(device))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).to(device), 
                                    torch.tensor(y_val, dtype=torch.float32).to(device))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=X_val.shape[0], shuffle=False)
        return train_loader, val_loader

    def _create_optimizer(self, optimizer_name: str, parameters, learning_rate: float, optimizer_params: Dict, l2_lambda: float):
        optimizers = {
            'Adam': optim.Adam,
            'AdamW': optim.AdamW,
        }
        optimizer_class = optimizers.get(optimizer_name)
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        return optimizer_class(parameters, lr=learning_rate, weight_decay=l2_lambda, **optimizer_params)


def plot_losses(loss_hist_train: List[float], loss_hist_valid: List[float]):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_hist_train, label='Training Loss')
    plt.plot(loss_hist_valid, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_model(trained_model: nn.Module, X_test: np.ndarray, device: str) -> np.ndarray:
    trained_model.to(device)
    trained_model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = trained_model(X_tensor).cpu().numpy()
    return predictions

def predict(trained_model: nn.Module, X: np.ndarray, scaler_X: MinMaxScaler, scaler_y: MinMaxScaler, device: str) -> np.ndarray:
    trained_model.to(device)
    trained_model.eval()
    X_normalized = scaler_X.transform(X)
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = trained_model(X_tensor).cpu().numpy()
    return scaler_y.inverse_transform(predictions)
