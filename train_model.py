import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the processed data
data = pd.read_csv('daily_traffic_by_boro.csv')

# Convert date to numerical features (e.g., day of the week, month)
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month

# Features and target variable
X = data[['DayOfWeek', 'Month']]  # Add more features if available
y = data['Vol']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Save the trained model
joblib.dump(model, 'traffic_volume_model.pkl')
print("Model saved as 'traffic_volume_model.pkl'")
