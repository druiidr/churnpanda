import pandas as pd
import glob

# Path to the CSV files (adjust the path if necessary)
csv_files = glob.glob(r'D:\churnai\rawdata/*.csv')

# Read and concatenate all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
data = pd.concat(df_list, ignore_index=True)

# Define the number of days of inactivity to consider a player as churned
DAYS_INACTIVITY_THRESHOLD = 5

# Identify churned players
# Assuming 'days_since_last_activity' is the column indicating inactivity
churned_players = data[data['days_since_last_activity'] >= DAYS_INACTIVITY_THRESHOLD]
active_players = data[data['days_since_last_activity'] < DAYS_INACTIVITY_THRESHOLD]

# Save the dataframes to new CSV files
churned_players.to_csv(r'D:\churnai\pocessedData\churned_players1.csv', index=False)
active_players.to_csv(r'D:\churnai\pocessedData\active_players1.csv', index=False)

print("Churned players and active players data have been saved successfully.")
