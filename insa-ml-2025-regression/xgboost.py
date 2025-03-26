import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

def xgboost_model(data, target_column):
    """
    Fonction qui applique un modèle de régression XGBoost sur un dataset.
    
    :param data: DataFrame contenant les caractéristiques du véhicule et la cible
    :param target_column: Le nom de la colonne cible à prédire (ici, les émissions de CO2)
    :return: Prédictions du modèle et erreur absolue moyenne (MAE)
    """
    
    # Séparation des caractéristiques (X) et de la cible (y)
    X = data.drop(columns=[target_column])  # Toutes les colonnes sauf la colonne cible
    y = data[target_column]  # La colonne cible (émissions de CO2)
    
    # Diviser les données en ensemble d'entraînement et de test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardisation des données (facultatif mais recommandé pour certains modèles comme XGBoost)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Création et entraînement du modèle XGBoost
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test_scaled)
    
    # Calcul de l'Erreur Absolue Moyenne (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    
    return y_pred, mae
