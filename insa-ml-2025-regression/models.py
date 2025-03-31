import numpy as np
import pandas as pd
import json

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim

class LinearRegressionModel: 
    def __init__(self):
        self.model = LinearRegression()
        self.mean_absolute_error = None

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
    def __init__(self, input_size, config_path):
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
            layers.append(nn.ReLu())  # Activation function
            prev_size = neurons
        
        layers.append(nn.Linear(prev_size, 1))  # Output layer for regression
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def fit(self, X_train, Y_train, epochs=1000, learning_rate=0.01, step_size=500, gamma=0.9):
        """
        Train the neural network with dynamic learning rate.
        
        :param X_train: Training feature matrix
        :param Y_train: Target vector
        :param epochs: Number of epochs to train
        :param learning_rate: Initial learning rate
        :param step_size: Number of epochs after which LR is reduced
        :param gamma: Multiplicative factor to reduce LR
        """
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        Y_train = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Dynamic Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.forward(X_train)
            loss = criterion(predictions, Y_train)
            loss.backward()
            optimizer.step()
            
            # Update Learning Rate
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()}')


    def test(self, X_test, Y_test):
        """Evaluate the model and return MAE."""
        X_test = torch.tensor(X_test, dtype=torch.float32)
        Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)
        
        with torch.no_grad():
            predictions = self.forward(X_test)
        
        mae = mean_absolute_error(Y_test.numpy(), predictions.numpy())
        return mae

    def predict(self, X):
        """Make predictions on new data."""
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(X).numpy()

    



        
