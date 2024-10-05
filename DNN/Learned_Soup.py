import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict

def get_model_weights_and_biases(model):
    weights = []
    biases = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'weight' in name:
                weights.append(param.data.clone())
            elif 'bias' in name:
                biases.append(param.data.clone())
    return weights, biases

class CustomANN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_activation, output_activation):
        super(CustomANN, self).__init__()
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # Create a list of layers
        layers = []
        all_layers = [input_size] + hidden_layers + [output_size]
        for i in range(len(all_layers) - 1):
            layers.append(nn.Linear(all_layers[i], all_layers[i + 1]))
            if i < len(all_layers) - 2:  # Apply activation function for hidden layers
                if hidden_activation != 'linear':
                    layers.append(getattr(nn, hidden_activation)())
        
        # Add output activation if it's not linear
        if output_activation != 'linear':
            layers.append(getattr(nn, output_activation)())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class AlphaWrapper(nn.Module):
    def __init__(self, models, model_args, device, use_zeta=False):
        super(AlphaWrapper, self).__init__()
        self.models = models
        self.alpha = nn.Parameter(torch.ones(len(models)) / len(models))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.model_args = model_args
        self.device = device
        self.use_zeta = use_zeta

        if use_zeta:
            self.zeta = nn.Parameter(torch.ones(len(models)) / len(models))

        weights = []
        biases = []
        
        for model in models:
            weights_i, biases_i = get_model_weights_and_biases(model)
            weights.append(weights_i)
            biases.append(biases_i)
        
        self.weights = weights
        self.biases = biases
        self.layers = self._build_layers()

    def _build_layers(self):
        layers = []
        input_size = self.model_args['input_size']
        hidden_layers = self.model_args['hidden_layers']
        output_size = self.model_args['output_size']

        # Define all layers based on the architecture
        all_layers = [input_size] + hidden_layers + [output_size]
        for i in range(len(all_layers) - 1):
            layers.append(nn.Linear(all_layers[i], all_layers[i + 1]))
        return nn.ModuleList(layers)

    def forward(self, x):
        alpha_softmax = F.softmax(self.alpha, dim=0)

        if self.use_zeta:
            zeta_softmax = F.softmax(self.zeta, dim=0)

        # Combine weights and biases using alpha and zeta (if applicable)
        combined_weights = []
        combined_biases = []
        for j in range(len(self.weights[0])):
            w_avg_ij = alpha_softmax[0] * self.weights[0][j]
            b_avg_ij = alpha_softmax[0] * self.biases[0][j] if not self.use_zeta else zeta_softmax[0] * self.biases[0][j]
            for i in range(1, len(self.models)):
                w_avg_ij += alpha_softmax[i] * self.weights[i][j]
                b_avg_ij += alpha_softmax[i] * self.biases[i][j] if not self.use_zeta else zeta_softmax[i] * self.biases[i][j]
            combined_weights.append(w_avg_ij)
            combined_biases.append(b_avg_ij)

        # Use the combined weights and biases directly in the forward pass
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = F.linear(x, combined_weights[idx], combined_biases[idx])
                if idx < len(self.layers) - 1:  # Apply activation function for hidden layers
                    if self.model_args['hidden_activation'] != 'linear':
                        x = getattr(F, self.model_args['hidden_activation'].lower())(x)
                else:
                    if self.model_args['output_activation'] != 'linear':
                        x = getattr(F, self.model_args['output_activation'].lower())(x)
        return self.beta * x

    def get_combined_model_and_params(self):
        alpha_softmax = F.softmax(self.alpha, dim=0)

        if self.use_zeta:
            zeta_softmax = F.softmax(self.zeta, dim=0)

        # Combine weights and biases using alpha and zeta (if applicable)
        combined_weights = []
        combined_biases = []
        for j in range(len(self.weights[0])):
            w_avg_ij = alpha_softmax[0] * self.weights[0][j]
            b_avg_ij = alpha_softmax[0] * self.biases[0][j] if not self.use_zeta else zeta_softmax[0] * self.biases[0][j]
            for i in range(1, len(self.models)):
                w_avg_ij += alpha_softmax[i] * self.weights[i][j]
                b_avg_ij += alpha_softmax[i] * self.biases[i][j] if not self.use_zeta else zeta_softmax[i] * self.biases[i][j]
            combined_weights.append(w_avg_ij)
            combined_biases.append(b_avg_ij)

        combined_model = construct_model(self.model_args, combined_weights, combined_biases, self.device)
        return combined_model, combined_weights, combined_biases

def construct_model(model_args, weights, biases, device):
    input_size = model_args['input_size']
    output_size = model_args['output_size']
    hidden_layers = model_args['hidden_layers']
    hidden_activation = model_args['hidden_activation']
    output_activation = model_args['output_activation']

    model = CustomANN(input_size, output_size, hidden_layers, hidden_activation, output_activation).to(device)
    
    with torch.no_grad():
        idx = 0
        for layer in model.model:
            if isinstance(layer, nn.Linear):
                layer.weight.data = weights[idx].to(device)
                layer.bias.data = biases[idx].to(device)
                idx += 1
    
    return model

def optimize_learned_soup(models, model_args, train_loader, epochs=128, lr=0.01, use_zeta=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alpha_model = AlphaWrapper(models, model_args, device, use_zeta=use_zeta).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(alpha_model.parameters(), lr=lr)
    
    loss_values = []

    for epoch in range(epochs):
        alpha_model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = alpha_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            loss_values.append(loss.item())

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    alpha_model.eval()
    return alpha_model.alpha.detach(), alpha_model.zeta.detach() if use_zeta else None, alpha_model.beta.detach(), alpha_model.get_combined_model_and_params(), loss_values

class LearnedSoup:
    def __init__(self, model_args: Dict, models: List[nn.Module], train_loader: DataLoader):
        self.model_args = model_args
        self.models = models
        self.train_loader = train_loader

    def optimize(self, epochs=128, lr=0.01, use_zeta=False):
        alpha, zeta, beta, (combined_model, combined_weights, combined_biases), loss_values = optimize_learned_soup(
            self.models, self.model_args, self.train_loader, epochs, lr, use_zeta
        )

        # Plot the loss values
        plt.plot(loss_values)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss per Batch')
        plt.show()

        return alpha, zeta, beta, combined_model, loss_values
