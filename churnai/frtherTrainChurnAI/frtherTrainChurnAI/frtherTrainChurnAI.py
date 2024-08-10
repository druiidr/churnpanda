#further train an existing model

import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Path to the churned and active player CSV files (adjust the paths if necessary)
churned_players_file = r'D:\churnai\pocessedData\churned_players1.csv'
active_players_file = r'D:\churnai\pocessedData\active_players1.csv'

# Read the CSV files
churned_players = pd.read_csv(churned_players_file)
active_players = pd.read_csv(active_players_file)

# Label the data
churned_players['churn_label'] = 1
active_players['churn_label'] = 0

# Combine the data
data = pd.concat([churned_players, active_players], ignore_index=True)

# Drop the user_id_str and days_since_last_activity columns
data = data.drop(columns=['player_id_str', 'days_since_last_activity'])

# Features and target
X = data.drop(columns=['churn_label'])
y = data['churn_label']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Load the existing model
model_path = r'D:\churnai\churn detect ai\churn detect ai\churn_prediction_model2.0.pkl'
pipeline = joblib.load(model_path)

# Split the new data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further train the existing model
pipeline.fit(X_train, y_train)

# Predict and evaluate on the new test set
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Display evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")

# (Optional) Perform hyperparameter tuning on the updated model
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Save the best model if tuning was performed
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Save the further trained model
joblib.dump(best_model, model_path)
print(f"Model updated and saved at {model_path}")

