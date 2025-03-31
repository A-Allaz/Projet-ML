import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from preprocess_data import preprocess_data_target, explore_data

# Fonction de classification avec Gradient Boosting
def gradient_boosting_classification(X_train, y_train, X_test):
    # Initialiser le modèle Gradient Boosting
    model = GradientBoostingClassifier(n_estimators=10000, random_state=42)

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    return model, y_pred

# Evaluation du modèle
def evaluate_model(y_true, y_pred):
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
    plt.title('Matrice de confusion')
    plt.ylabel('Vrai')
    plt.xlabel('Prédit')
    plt.show()

    # Rapport de classification
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

# Charger et prétraiter les données
fileToClean = 'data/train.csv'
cleanedFile = 'data/cleaned_train.csv'
preprocess_data_target(fileToClean, cleanedFile)
df = pd.read_csv(cleanedFile)
explore_data(df)

# Séparation des features et de la cible
X = df.drop(columns=['bc_price_evo'])  # Supposons que 'bc_price_evo' est la cible
y = df['bc_price_evo']

# Séparation en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Définir le modèle de base
# gbc = GradientBoostingClassifier(random_state=42)

# # Définir l'espace des hyperparamètres à tester
# param_grid = {
#     'n_estimators': [100, 200, 300, 400, 500],  # Tester différentes valeurs de n_estimators
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Tester différentes valeurs de taux d'apprentissage
#     'max_depth': [3, 5, 10],  # Tester différentes profondeurs d'arbres
#     'min_samples_split': [2, 5, 10],  # Tester différentes tailles de sous-échantillons pour le split
#     'min_samples_leaf': [1, 2, 4]  # Tester différentes tailles minimales de feuilles
# }

# # Initialiser le GridSearchCV avec validation croisée
# grid_search = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# # Lancer la recherche des meilleurs hyperparamètres
# grid_search.fit(X_train, y_train)

# # Afficher les meilleurs paramètres
# print("Meilleurs hyperparamètres trouvés :", grid_search.best_params_)

# # Prédictions avec le meilleur modèle
# best_gbc_model = grid_search.best_estimator_
# y_pred = best_gbc_model.predict(X_test)

# # Calculer l'accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy sur le jeu de test avec les meilleurs hyperparamètres : {accuracy * 100:.2f}%")

# Entraînement et évaluation du modèle
model, y_pred = gradient_boosting_classification(X_train, y_train, X_test)
evaluate_model(y_test, y_pred)

# --- Nouveau code pour la soumission ---

# Charger les données de test (données réelles pour la soumission)
test_file = 'data/test.csv'
test_df = pd.read_csv(test_file)

# Effectuer les prédictions sur l'ensemble de test
X_test = test_df.drop(columns=['id'])  # Ne pas inclure 'id' dans les features
test_predictions = model.predict(X_test)

# Convertir les prédictions en 'UP' ou 'DOWN'
test_predictions = ['UP' if p == 1 else 'DOWN' for p in test_predictions]

# Sauvegarder les résultats dans un fichier de soumission
def save_submission(predictions, test_ids, output_file):
    submission_df = pd.DataFrame({
        'id': test_ids,
        'bc_price_evo': predictions
    })
    submission_df.to_csv(output_file, index=False)

# Sauvegarder le fichier de soumission avec les prédictions
save_submission(test_predictions, test_df['id'], 'submission_data/submission3.csv')
