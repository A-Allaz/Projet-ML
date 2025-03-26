import numpy as np
import sklearn
import pandas as pd
import csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

with open("data/train.csv") as my_csv_file:
  train_dataset = list(csv.reader(my_csv_file, delimiter=','))

train_header = train_dataset[0]
train_data = train_dataset[1:]

data_array = np.array(train_data)
# Identifier les indices des colonnes numériques
data = pd.read_csv("data/train.csv")
string_columns = data.select_dtypes(include=['number']).columns
numerical_indices = [data.columns.get_loc(col) for col in string_columns]

# Extraire les données numériques
numerical_data = data_array[:, numerical_indices].astype(float)

# Séparer X et y
X = numerical_data[:, :-1]  # Toutes les colonnes sauf "co2"
y = numerical_data[:, -1]   # "co2"

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entraîner le modèle
model = LinearRegression()
model.fit(X_scaled, y)

# Prédire les émissions de CO2
X_test = np.array([[100, 1000, 2000, 5, 6, 7]])
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)




# https://www.youtube.com/watch?v=dQw4w9WgXcQ

