import numpy as np
import pandas as pd
import json

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LinearRegressionModel: 
    def __init__(self):
        self.model = LinearRegression()
        self.mean_absolute_error = None
        device = "cpu" # Default to CPU
        if torch.cuda.is_available():
            device = "cuda" # Use NVIDIA GPU (if available)
        elif torch.backends.mps.is_available():
            device = "mps" # Use Apple Silicon GPU (if available)

        self.device = device
        print(f"Using {self.device} device for model")

    def fit(self, X, Y):
        """
        Fit the linear regression model to the training data.
        
        :param X: Feature matrix
        :param y: Target vector
        """
        self.model.fit(X, Y)

    def test(self, X, Y):
        """
        Test the linear regression model.
        
        :param X: Feature matrix
        :param y: Target vector
        :return: Mean Absolute Error (MAE)
        """
        Y_pred = self.model.predict(X)
        self.mean_absolute_error = mean_absolute_error(Y, Y_pred)
        return self.mean_absolute_error
        

    def predict(self, X):
        """
        Predict using the linear regression model.
        
        :param X: Feature matrix
        :return: Predicted values
        """

        y_pred = self.model.predict(X)
        return y_pred

class MultiLayerFeedForwardModel(nn.Module):
    def __init__(self, input_size, config_path, save_path: str = None):
        """
        :param input_size: Number of input features
        :param config: List of integers specifying the number of neurons per layer
        """
        super(MultiLayerFeedForwardModel, self).__init__()
        layers = []
        prev_size = input_size

        # Load the configuration from the file
        with open(config_path, "r") as f:
            self.config = json.load(f)["layers"]
        
        for neurons in self.config:
            layers.append(nn.Linear(prev_size, neurons))
            layers.append(nn.ReLU())  # Activation function
            prev_size = neurons
        
        layers.append(nn.Linear(prev_size, 1))  # Output layer for regression
        self.model = nn.Sequential(*layers)

        device = "cpu" # Default to CPU
        if torch.cuda.is_available():
            device = "cuda" # Use NVIDIA GPU (if available)
        elif torch.backends.mps.is_available():
            device = "mps" # Use Apple Silicon GPU (if available)

        self.device = device
        self.to(self.device)  # Move model to the specified device
        print(f"Using {self.device} device for model")
        self.save_path = save_path

    def forward(self, x):
        return self.model(x)
    
    def fit(self,
        X_train,
        Y_train,
        X_val,
        Y_val,
        epochs=2000,
        learning_rate=0.01,
        step_size=100,
        gamma=0.9,
        batch_size=32,
        patience=None,
        store_improvement: bool=False,
    ):

        """
        Train the neural network with dynamic learning rate and early stopping.

        :param X_train: Training feature matrix
        :param Y_train: Training target vector
        :param X_val: Validation feature matrix
        :param Y_val: Validation target vector
        :param epochs: Number of epochs to train
        :param learning_rate: Initial learning rate
        :param step_size: Number of epochs after which LR is reduced
        :param gamma: Multiplicative factor to reduce LR
        :param patience: Number of epochs to wait for improvement before stopping
        """
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        Y_val = torch.tensor(Y_val, dtype=torch.float32).view(-1, 1).to(self.device)

        dataset = TensorDataset(X_train, Y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Dynamic Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.train()  # Set the model to training mode
            for batch_X, batch_Y in dataloader:
                optimizer.zero_grad()
                predictions = self.forward(batch_X)
                loss = criterion(predictions, batch_Y)
                loss.backward()
                optimizer.step()

            # Update Learning Rate
            scheduler.step()


            if (epoch + 1) % 10 == 0:
                # Validation phase
                self.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    val_predictions = self.forward(X_val)
                    val_loss = criterion(val_predictions, Y_val).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if store_improvement and self.save_path is not None:
                        torch.save(self.state_dict(), self.save_path)  # Save the best model
                else:
                    patience_counter += 1
                if patience: # we now save the model if the validation loss is not improving
                    if patience_counter >= patience:
                        print(f'Early stopping at epoch {epoch+1}')
                        break
                    
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.8f}, Val Loss: {val_loss:.8f}, LR: {scheduler.get_last_lr()}')



    def test(self, X_test, Y_test):
        """Evaluate the model and return MAE."""

        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1).to(self.device)

        with torch.no_grad():
            predictions = self.forward(X_test)

        # Move tensors to CPU before converting to NumPy
        mae = mean_absolute_error(Y_test.cpu().numpy(), predictions.cpu().numpy())
        return mae


    def predict(self, X):
        """Make predictions on new data."""
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.forward(X).cpu().numpy()

    def store(self, path:str):
        """Store the model parameters."""
        torch.save(self.state_dict(), path)
        print(f'Model stored at {path}')
    
    def load(self, path:str):
        """Load the model parameters."""
        self.load_state_dict(torch.load(path))
        print(f'Model loaded from {path}')

