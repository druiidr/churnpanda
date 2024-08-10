#interpret a model

import pandas as pd
import joblib

# Path to the saved model (adjust the path if necessary)
model_path = r'D:\churnai\saved_models\churn_prediction_model2.0.pkl'

# Load the saved model
pipeline = joblib.load(model_path)

# Path to the new data or test set
new_data_file = r'D:\churnai\testdata\sneakypandatestraw.csv'

# Read the new data
new_data = pd.read_csv(new_data_file)

# Set the number of days to extend the data for prediction
x_days = 5

# Drop unnecessary columns except 'churned'
X_new = new_data.drop(columns=['player_id_str', 'days_since_last_activity'])

# Add a dummy 'churned' column with placeholder values
X_new['churned'] = 0  # This is just a placeholder

# Make current churn predictions
y_pred_current = pipeline.predict(X_new)
y_proba_current = pipeline.predict_proba(X_new)[:, 1]

# Simulate data for x days assuming zero activities
# Create a copy of the dataset for churn prediction within x days
X_future = X_new.copy()

# Set activity columns to zero for the future days
for day in range(16, 16 + x_days):
    for column in ['event_count_day', 'session_start_count_day', 'payment_count_day']:
        X_future[f'{column}{day}'] = 0

# Make future churn predictions (within x days)
y_pred_future = pipeline.predict(X_future)
y_proba_future = pipeline.predict_proba(X_future)[:, 1]

# Display the predictions and their corresponding probabilities
results = pd.DataFrame({
    'Player_ID': new_data['player_id_str'],
    'Current_Churn_Prediction': y_pred_current,
    'Current_Churn_Probability': y_proba_current,
    f'Churn_Probability_within_{x_days}_days': y_proba_future
})

# Print the results
print(results)

# Optionally, save the results to a CSV file
results.to_csv(r'D:\churnai\pocessedDataprediction_results1.csv', index=False)

print("Results have been saved to 'pocessedDataprediction_results1.csv'")
