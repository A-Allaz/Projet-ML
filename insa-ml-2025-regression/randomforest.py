from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import sys

import category_encoders as ce

def smooth_target_encoding(df, cat_cols, target_col, alpha=10):
    """
    Applique un Target Encoding avec smoothing pour éviter le sur-apprentissage.
    alpha : paramètre de lissage (plus il est grand, plus on se rapproche de la moyenne globale).
    """
    global_mean = df[target_col].mean()
    encoded_cols = {}

    for col in cat_cols:
        stats = df.groupby(col)[target_col].agg(['mean', 'count'])
        smooth = (stats['mean'] * stats['count'] + global_mean * alpha) / (stats['count'] + alpha)
        encoded_cols["encoded_"+col] = df[col].map(smooth)

    return df.assign(**encoded_cols)


def random_forest(df_train, target_column,hyperparameter=10):
    """
    Modèle de régression utilisant RandomForest avec encodage des variables catégorielles et imputation des valeurs manquantes.
    
    :param data: DataFrame contenant les caractéristiques du véhicule et la cible
    :param target_column: Le nom de la colonne cible à prédire (émissions de CO2)
    :return: Prédictions du modèle et erreur absolue moyenne (MAE)
    """
    
    # On récupère les colonnes dont les valeurs sont des chaînes de caractères
    string_columns = df_train.select_dtypes(include=['object']).columns

    # Détection des colonnes catégorielles et numériques
    categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
    
    df_train = smooth_target_encoding(df_train, categorical_cols, 'co2')

    #mettre la colonne hcnox à hc + nox si elle est nulle
    df_train['hcnox'] = df_train['hcnox'].fillna(df_train['hc'] + df_train['nox'])
    
    dropcols = ['id', 'co', 'hc', 'nox','ptcl','encoded_fuel_type','encoded_hybrid'] + [target_column] + categorical_cols


    # Séparation des features et de la cible
    X = df_train.drop(columns=dropcols)  # Toutes les colonnes sauf la colonne cible
    y = df_train[target_column]

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    #fill na avec la moyenne pour les colonnes numériques
    for col in numerical_cols:
        X[col] = X[col].fillna(X[col].mean())

    # Définition du modèle Random Forest
    model = RandomForestRegressor(n_estimators=hyperparameter, max_depth = None , min_samples_leaf=1)
    # todo : extra trees
    # Création du pipeline
    pipeline = Pipeline(steps=[
        ('regressor', model)
    ])
    
    # Séparation en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(y_test.head())
    model.fit(X_train, y_train)

    # Entraînement du modèle
    pipeline.fit(X_train, y_train)
    
    # Prédictions
    y_pred = pipeline.predict(X_test)
    
    # Calcul de l'erreur absolue moyenne (MAE)
    mae = mean_absolute_error(y_test, y_pred)

    import matplotlib.pyplot as plt

    feature_importances = pipeline.named_steps['regressor'].feature_importances_
    features = X_train.columns

    plt.figure(figsize=(10, 5))
    plt.barh(features, feature_importances)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importances")
    plt.show()

    
    return y_pred, mae, X_test.index


def random_forest_test(df_train, df_test, target_column,hyperparameter=10):
    """
    Modèle de régression utilisant RandomForest avec encodage des variables catégorielles et imputation des valeurs manquantes.
    
    :param data: DataFrame contenant les caractéristiques du véhicule et la cible
    :param target_column: Le nom de la colonne cible à prédire (émissions de CO2)
    :return: Prédictions du modèle et erreur absolue moyenne (MAE)
    """

    indexs = df_test["id"]

    # On récupère les colonnes dont les valeurs sont des chaînes de caractères
    string_columns = df_train.select_dtypes(include=['object']).columns

    # Détection des colonnes catégorielles et numériques
    categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
    
    df_train = smooth_target_encoding(df_train, categorical_cols, 'co2')

    #remplacer les valeurs catégoriques dans le dataframe de test par leur valeur encodée correspondante dans le dataframe d'entraînement
    for col in categorical_cols:
        df_test["encoded_"+col] = df_test[col].map(df_train.groupby(col)["co2"].mean())

    print(df_train.head())
    print(df_test.head())
    print(indexs)


    #mettre la colonne hcnox à hc + nox si elle est nulle
    df_train['hcnox'] = df_train['hcnox'].fillna(df_train['hc'] + df_train['nox'])
    df_test['hcnox'] = df_test['hcnox'].fillna(df_test['hc'] + df_test['nox'])
    
    
    dropcols = ['id', 'co', 'hc', 'nox','ptcl','encoded_fuel_type','encoded_hybrid'] + categorical_cols

    ###############
    import plotly.express as px
    import plotly.io as pio

    numeric_columns = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    correlation_matrix = df_train[numeric_columns].corr()
    correlation_matrix = correlation_matrix.reset_index().melt(id_vars='index')

    # Générer la heatmap (un peu) interactive
    fig = px.imshow(
        df_train[numeric_columns].corr(),
        text_auto=True,
        color_continuous_scale="RdBu",
        title="Matrice de Corrélation"
    )

    # Sauvegarde de la heatmap en HTML
    with open("./plots/correlation_matrix.html", "w", encoding="utf-8") as f:
        f.write(pio.to_html(fig, full_html=True, include_plotlyjs='cdn'))

    ###############

    # Séparation des features et de la cible
    X = df_train.drop(columns=dropcols)  # Toutes les colonnes sauf la colonne cible
    X.drop(columns=target_column, inplace=True)  
    y = df_train[target_column]

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Prétraitement des données : imputation + encodage des variables catégorielles
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numerical_cols)  # Remplacement des NaN numériques par la moyenne
        ],
        remainder='passthrough'
    )
    #fill na avec la moyenne pour les colonnes numériques
    for col in numerical_cols:
        X[col] = X[col].fillna(X[col].mean())

    # Définition du modèle Random Forest
    model = RandomForestRegressor(n_estimators=hyperparameter, max_depth = None , min_samples_leaf=1)
    # todo : extra trees

    # Création du pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
        
    # Entraînement du modèle
    pipeline.fit(X, y)
    
    # Prédictions
    y_pred = pipeline.predict(df_test.drop(columns=dropcols))
    
  
    return y_pred, indexs


# Charger les données
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/submission_test.csv')

# predictions,mae=random_forest_model_with_encoding(data, 'co2')
# submission = pd.DataFrame({
#     "id": data.index[:len(predictions)],
#     "co2": predictions
# })

if sys.argv[1] == 'test':
    predictions, test_indices = random_forest_test(df_train,df_test,'co2',10)
else:
    predictions, mae, test_indices = random_forest(df_train,'co2',200)
    print("Prédictions sur l'ensemble de test:", predictions[0:6])
    print('Erreur absolue moyenne:', mae)

# Créer un DataFrame pour les soumissions
submission = pd.DataFrame({
    "id": test_indices,  # Utilise les indices sauvegardés avant train_test_split
    "co2": predictions
})

submission.to_csv("submission_data/rainforest_prediction.csv", index=False)
print("Prédictions enregistrées dans 'submission_data/rainforest_prediction.csv'")

