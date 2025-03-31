import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def linear_regression_model_without_encoding(data, target_column):
    """
    Fonction qui applique un modèle de régression linéaire sur un dataset.
    
    :param data: DataFrame contenant les caractéristiques du véhicule et la cible
    :param target_column: Le nom de la colonne cible à prédire (ici, les émissions de CO2)
    :return: Prédictions du modèle et erreur absolue moyenne (MAE)
    """
    
    # Séparation des caractéristiques (X) et de la cible (y)
    X = data.drop(columns=[target_column])  # Toutes les colonnes sauf la colonne cible
    y = data[target_column]  # La colonne cible (émissions de CO2)
    
    # Diviser les données en ensemble d'entraînement et de test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardisation des données (très important pour la régression linéaire)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Création et entraînement du modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test_scaled)
    
    # Calcul de l'Erreur Absolue Moyenne (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    
    return y_pred, mae


def linear_regression_model_with_encoding(data, target_column):
    """
    Fonction qui applique un modèle de régression linéaire sur un dataset avec encodage des variables catégorielles.
    
    :param data: DataFrame contenant les caractéristiques du véhicule et la cible
    :param target_column: Le nom de la colonne cible à prédire (ici, les émissions de CO2)
    :return: Prédictions du modèle et erreur absolue moyenne (MAE)
    """
    
    # Séparation des caractéristiques (X) et de la cible (y)
    X = data.drop(columns=[target_column])  # Toutes les colonnes sauf la colonne cible
    y = data[target_column]  # La colonne cible (émissions de CO2)
    
    # Liste des colonnes catégorielles à encoder
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Création d'un préprocesseur avec OneHotEncoder pour les colonnes catégorielles
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols)  # Appliquer OneHotEncoder aux colonnes catégorielles
        ], 
        remainder='passthrough'  # Conserver les autres colonnes numériques sans modification
    )
    
    # Création d'un pipeline avec prétraitement et modèle de régression linéaire
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Diviser les données en ensemble d'entraînement et de test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraînement du modèle avec les données de formation
    pipeline.fit(X_train, y_train)
    
    # Prédictions sur l'ensemble de test
    y_pred = pipeline.predict(X_test)
    
    # Calcul de l'Erreur Absolue Moyenne (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    
    return y_pred, mae

if __name__ == "__main__":
    # Appliquer le modèle avec encodage
    # Charger les données
    
    data = pd.read_csv('data/train.csv')
    predictions,mae=linear_regression_model_with_encoding(data, 'co2')
    print("Prédictions sur l'ensemble de test:", predictions)
    print('Erreur absolue moyenne:', mae)
