import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data_target(data_path, save_path):
    # Charger les données
    data = pd.read_csv(data_path)
    
    # Séparer les colonnes numériques et catégorielles
    # numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    # Remplacer les valeurs manquantes uniquement pour les colonnes numériques
    # Exclure la colonne bc_price_evo de l'opération fillna
    # data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    
    # Encodage de la variable cible (bc_price_evo)
    data['bc_price_evo'] = data['bc_price_evo'].map({'UP': 1, 'DOWN': 0})
    
    
    # Suppression des colonnes inutiles
    data = data.drop(columns=['id'])
    
    # Sauvegarder le fichier prétraité
    data.to_csv(save_path, index=False)
    return save_path


fileToClean = 'data/train.csv'
cleanedFile = 'data/cleaned_train.csv'
preprocess_data_target(fileToClean, cleanedFile)


# Exploration des relations entre les variables
def explore_data(df):
    # Matrice de corrélation
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Matrice de corrélation des variables")
    plt.savefig("output_pictures/correlation_matrix.png")
    plt.close() 

    # Distribution de la variable cible
    sns.countplot(x=df['bc_price_evo'])
    plt.title("Distribution de la variable cible")
    plt.savefig("output_pictures/target_distribution.png") 
    plt.close()

    # Relation entre bc_price et bc_price_evo
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['bc_price_evo'], y=df['bc_price'])
    plt.title("Relation entre bc_price et bc_price_evo")
    plt.savefig("output_pictures/bc_price_vs_bc_price_evo.png")
    plt.close()

    # Relation entre bc_demand et bc_price_evo
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['bc_price_evo'], y=df['bc_demand'])
    plt.title("Relation entre bc_demand et bc_price_evo")
    plt.savefig("output_pictures/bc_demand_vs_bc_price_evo.png")
    plt.close()


df = pd.read_csv(cleanedFile)
explore_data(df)