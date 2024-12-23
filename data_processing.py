import pandas as pd

# Load the dataset
data = pd.read_csv('Automated_Traffic_Volume_Counts.csv')

# Print the column names to check what's available
print("Columns in dataset:", data.columns)

# Create a 'Date' column by combining Yr, M, D columns into a string and converting to datetime
data['Date'] = pd.to_datetime(data['Yr'].astype(str) + '-' + 
                               data['M'].astype(str).str.zfill(2) + '-' + 
                               data['D'].astype(str).str.zfill(2))

# Preprocess the traffic volume data
data['Vol'] = data['Vol'].astype(float)  # Ensure the volume is of type float

# Group by Date and Boro (Borough) to get daily traffic volume per borough
daily_traffic = data.groupby(['Date', 'Boro']).agg({'Vol': 'sum'}).reset_index()

# Calculate average traffic volume per borough
avg_traffic_by_boro = daily_traffic.groupby('Boro')['Vol'].mean()

# Output the results to a CSV file
daily_traffic.to_csv('daily_traffic_by_boro.csv', index=False)
avg_traffic_by_boro.to_csv('avg_traffic_by_boro.csv')

print(daily_traffic.head())
print(avg_traffic_by_boro)