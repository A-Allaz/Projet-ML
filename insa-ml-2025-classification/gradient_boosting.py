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



# Initialize the Gradient Boosting Classifier
model = GradientBoostingClassifier(n_estimators=5000, min_samples_split=2, min_samples_leaf=2, max_depth=6, learning_rate=0.45)
# --> {'learning_rate': 0.3, 'max_depth': 6, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 1500} for an accuracy of .9233
#après test : {'learning_rate': 0.45, 'max_depth': 6, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 5000} for an accuracy of .9382


# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

y_pred= ['UP' if p ==1 else 'DOWN' for p in y_pred]

# Output the predictions
predictions_df = test_df[['id']].copy()
predictions_df['bc_price_evo'] = y_pred


# Save the predictions to a CSV file
predictions_df.to_csv('./submission_data/prediction2.csv', index=False)


# Assuming 'model' is your trained GradientBoostingClassifier
model_params = model.get_params()
# Print the model's parameters
print("Model Parameters:")
for param, value in model_params.items():
    print(f"{param}: {value}")

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))



