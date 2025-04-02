import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Load the datasets
train_df = pd.read_csv('./data/cleaned_train.csv')
test_df = pd.read_csv('./data/test.csv')

# Define features and target variable
features = ['date', 'hour', 'bc_price', 'bc_demand', 'ab_price', 'ab_demand', 'transfer']
target = 'bc_price_evo'

# Split the training data into features and target
X_train = train_df[features]
y_train = train_df[target]

# Split the test data into features (no target variable in test set)
X_test = test_df[features]


# Define the parameter grid
param_grid = {
    'n_estimators': [300, 500, 1000, 1500],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 5]
}

model = GradientBoostingClassifier(random_state=42)
""" # Set up Random Search with cross-validation

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)

# Fit Random Search to the data
random_search.fit(X_train, y_train)

# Best parameters and best score
best_params = random_search.best_params_
best_score = random_search.best_score_  """

# Set up Grid Search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# --> {'learning_rate': 0.3, 'max_depth': 6, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 1500} for an accuracy of .9233

# Fit Grid Search to the data
grid_search.fit(X_train, y_train)

# Best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f'Best Parameters: {best_params}')
print(f'Best Cross-Validation Accuracy: {best_score}')
