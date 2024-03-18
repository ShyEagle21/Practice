import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

torch.manual_seed(0)

class NN_Builder:
    def __init__(self, csv_file, batch_size=128, num_layers_list=[1,3,5,10], num_nodes_list=[1,5,10,100,1000], activation_functions=["relu"], epochs=50, learning_rate=0.01):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.num_layers_list = num_layers_list
        self.num_nodes_list = num_nodes_list
        self.activation_functions = activation_functions
        self.epochs = epochs
        self.learning_rate = learning_rate

        # Read data from CSV
        df = pd.read_csv(csv_file)
        #must be updated based on the data set for what the target variable is
        #######################################################################################
        y = pd.CategoricalIndex(df.travel_mode).codes
        X = df.loc[:, df.columns != 'travel_mode'].to_numpy()
        #######################################################################################
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
        self.train_dataset = TensorDataset(Tensor(X_train), Tensor(y_train))
        self.val_dataset = TensorDataset(Tensor(X_val), Tensor(y_val))
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

    class NeuralNetwork(nn.Module):
        def __init__(self, num_layers, num_nodes, activation):
            super().__init__()
            layers = []
            layers.append(nn.Linear(in_features=22, out_features=num_nodes))
            if activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            else:
                raise ValueError("Unsupported activation function")
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(in_features=num_nodes, out_features=num_nodes))
                if activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                else:
                    raise ValueError("Unsupported activation function")
            layers.append(nn.Linear(in_features=num_nodes, out_features=4))
            self.linear_relu_stack = nn.Sequential(*layers)

        def forward(self, x):
            return self.linear_relu_stack(x)

    def train_model(self, model, optimizer, criterion, train_loader, val_loader, epochs):
        train_acc_history = []
        val_acc_history = []
        for epoch in range(epochs):
            train_acc = 0.0
            for X, y in train_loader:
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y.long())
                loss.backward()
                optimizer.step()
                train_acc += (pred.softmax(dim=1).argmax(dim=1) == y).type(torch.float).sum() / len(self.train_dataset)
            train_acc_history.append(train_acc)

            val_acc = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    pred = model(X)
                    val_acc += (pred.argmax(dim=1) == y).type(torch.float).mean().item()
                val_acc /= len(val_loader)
                val_acc_history.append(val_acc)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1} train accuracy: {train_acc:.4f}, validation accuracy: {val_acc:.4f}")
        return train_acc_history, val_acc_history

    def build_networks(self):
        for num_layers in self.num_layers_list:
            for num_nodes in self.num_nodes_list:
                for activation in self.activation_functions:
                    # Construct neural network
                    model = self.NeuralNetwork(num_layers, num_nodes, activation)
                    optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
                    criterion = nn.CrossEntropyLoss()

                    # Train and evaluate the model
                    self.train_model(model, optimizer, criterion, self.train_loader, self.val_loader, self.epochs)

# Define hyperparameters
csv_file = 'your_data.csv'
batch_size = 128
epochs = 30
learning_rate = 0.001
num_layers_list = [3]
num_nodes_list = [1000]
activation_functions = ['relu']

# Instantiate the NN_Builder class
nn_builder = NN_Builder(csv_file=csv_file, batch_size=batch_size, num_layers_list=num_layers_list, num_nodes_list=num_nodes_list, activation_functions=activation_functions, epochs=epochs, learning_rate=learning_rate)
nn_builder.build_networks()
