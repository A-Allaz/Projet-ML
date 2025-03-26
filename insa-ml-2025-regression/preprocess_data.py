import pandas as pd
import numpy as np
import os

def preprocess_data_target(data_path, save_path):
    data = pd.read_csv(data_path)

    # On remplace les valeurs manquantes par la moyenne de la colonne
    #data = data.fillna(data.mean())

    # On récupère les colonnes dont les valeurs sont des chaînes de caractères
    string_columns = data.select_dtypes(include=['object']).columns
    
    # On récupère les colonnes qui ont plus de 30 valeurs uniques
    high_cardinality_columns = [col for col in string_columns if data[col].nunique() > 1]
    print(high_cardinality_columns)
    # On Récupère la moyenne d’émission de CO2 pour chaque valeur unique des colonnes à haute cardinalité
    mean_co2_per_value = {}
    for col in high_cardinality_columns:
        mean_co2_per_value[col] = data.groupby(col)['co2'].mean()
    print(mean_co2_per_value)
    # On remplace les valeurs des colonnes à haute cardinalité par la moyenne d’émission de CO2 correspondante
    for col in high_cardinality_columns:
        data[col] = data[col].map(mean_co2_per_value[col])

    # On sauvegarde le fichier
    data.to_csv(save_path, index=False)
    return save_path

fileToClean = 'data/train.csv'
cleanedFile = 'data/cleaned_train.csv'
preprocess_data_target(fileToClean, cleanedFile)