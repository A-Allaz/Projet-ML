import pandas as pd
import numpy as np
import os

class PreProcessingData:
    def __init__(self, data_path: str, submission_data_path: str):
        self.data_path = data_path
        self.submission_data_path = submission_data_path

        # Load the data
        self.__load_data__()

    def __load_data__(self):
        pass

    def drop_low_correlation_columns(self):
        pass

    def drop_text_columns(self):
        pass

    def normalisation(self):
        pass

    def combine_hc_nox(self):
        pass

    def smooth_target_encoding(self):
        pass

    def fill_na_values(self):
        pass

    def change_low_number_of_categories(self):
        pass

def preprocess_data_target(data_path, submission_data_path, normalisation, save_path:str = None):
    data = pd.read_csv(data_path)
    submission_data = pd.read_csv(submission_data_path)


    # On récupère les colonnes dont les valeurs sont des chaînes de caractères
    string_columns = data.select_dtypes(include=['object']).columns
    
    # On récupère les colonnes qui ont plus d’une valeur unique
    high_cardinality_columns = [col for col in string_columns if data[col].nunique() > 1]


    # On Récupère la moyenne d’émission de CO2 pour chaque valeur unique des colonnes à haute cardinalité
    mean_co2_per_value = {}
    for col in high_cardinality_columns:
        mean_co2_per_value[col] = data.groupby(col)['co2'].mean()

    # On remplace les valeurs des colonnes à haute cardinalité par la moyenne d’émission de CO2 correspondante
    for col in high_cardinality_columns:
        data[col] = data[col].map(mean_co2_per_value[col])
        submission_data[col] = submission_data[col].map(mean_co2_per_value[col])

    # On remplace les valeurs manquantes par la moyenne de la colonne
    data = data.fillna(data.mean())
    submission_data = submission_data.fillna(data.mean())


    min_co2 = data['co2'].min()
    max_co2 = data['co2'].max()



    # On normalise les colonnes en mettant à l’échelle les valeurs entre 0 et 1
    if normalisation:
        for col in data.columns.drop(['id']):
            if col != 'co2':
                submission_data[col] = (submission_data[col] - data[col].min()) / (data[col].max() - data[col].min())
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
            
    
    # On sauvegarde le fichier
    if save_path:
        data.to_csv(save_path, index=False)
        submission_data.to_csv(save_path.replace('.csv', '_submission.csv'), index=False)

    return data, submission_data, min_co2, max_co2


if __name__ == '__main__':
    fileToClean = 'data/train.csv'
    cleanedFile = 'data/cleaned_train.csv'
    preprocess_data_target(fileToClean, cleanedFile)