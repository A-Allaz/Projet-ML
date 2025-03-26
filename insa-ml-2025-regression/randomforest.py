from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

def random_forest_model_with_encoding(data, target_column):
    """
    Modèle de régression utilisant RandomForest avec encodage des variables catégorielles et imputation des valeurs manquantes.
    
    :param data: DataFrame contenant les caractéristiques du véhicule et la cible
    :param target_column: Le nom de la colonne cible à prédire (émissions de CO2)
    :return: Prédictions du modèle et erreur absolue moyenne (MAE)
    """
    
    # Séparation des features et de la cible
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Détection des colonnes catégorielles et numériques
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Prétraitement des données : imputation + encodage des variables catégorielles
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Remplacer les NaN par la valeur la plus fréquente
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols),
            ('num', SimpleImputer(strategy='mean'), numerical_cols)  # Remplacement des NaN numériques par la moyenne
        ],
        remainder='passthrough'
    )
    
    # Définition du modèle Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Création du pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Séparation en ensemble d'entraînement et de test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraînement du modèle
    pipeline.fit(X_train, y_train)
    
    # Prédictions
    y_pred = pipeline.predict(X_test)
    
    # Calcul de l'erreur absolue moyenne (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    
    return y_pred, mae

def random_forest_model_with_encoding2(data, target_column):
    """
    Modèle de régression utilisant RandomForest avec encodage des variables catégorielles et imputation des valeurs manquantes.
    
    :param data: DataFrame contenant les caractéristiques du véhicule et la cible
    :param target_column: Le nom de la colonne cible à prédire (émissions de CO2)
    :return: Prédictions du modèle, erreur absolue moyenne (MAE) et indices des données test
    """
    
    # Séparation des features et de la cible
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Sauvegarde des indices d'origine
    X_indices = X.index
    
    # Détection des colonnes catégorielles et numériques
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Prétraitement des données : imputation + encodage des variables catégorielles
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Remplacer les NaN par la valeur la plus fréquente
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols),
            ('num', SimpleImputer(strategy='mean'), numerical_cols)  # Remplacement des NaN numériques par la moyenne
        ],
        remainder='passthrough'
    )
    
    # Définition du modèle Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Création du pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Séparation en ensemble d'entraînement et de test (80% train, 20% test)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, X_indices, test_size=0.2, random_state=42
    )
    
    # Entraînement du modèle
    pipeline.fit(X_train, y_train)
    
    # Prédictions
    y_pred = pipeline.predict(X_test)
    
    # Calcul de l'erreur absolue moyenne (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    
    return y_pred, mae, idx_test

# Charger les données
data = pd.read_csv('data/train.csv')
# predictions,mae=random_forest_model_with_encoding(data, 'co2')
# submission = pd.DataFrame({
#     "id": data.index[:len(predictions)],
#     "co2": predictions
# })
# Vérifie que la colonne 'id' existe sinon utilise l'index
if 'id' in data.columns:
    data.set_index('id', inplace=True)

predictions, mae, test_indices = random_forest_model_with_encoding2(data, 'co2')

# Créer un DataFrame pour les soumissions
submission = pd.DataFrame({
    "id": test_indices,  # Utilise les indices sauvegardés avant train_test_split
    "co2": predictions
})

submission.to_csv("submission_data/rainforest_prediction.csv", index=False)
print("Prédictions enregistrées dans 'submission_data/rainforest_prediction.csv'")
print("Prédictions sur l'ensemble de test:", predictions)
print('Erreur absolue moyenne:', mae)