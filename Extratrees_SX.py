import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# Load data from Excel file
# File path to the database goes here
file_path = "file_path_goes_here"
df = pd.read_excel(file_path)

# Ignore the column named "Reference (DOI)"
df = df.drop(columns=["Reference (DOI)"])

# Filter out extraction efficiency values lower than 1% and higher than 99%
df = df[(df['E'] <= 0.99) & (df['E'] >= 0.01)]

# Preprocessing
features = df.drop(columns=["E"])
target = df["E"]

# Remove Outliers using Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(features)

# Filter out the outliers
features_clean = features[outliers != -1]
target_clean = target[outliers != -1]

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(features_clean, target_clean, test_size=0.3, random_state=42)

# Scaling features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize optimised ExtraTreesRegressor 
extra_trees = ExtraTreesRegressor(
    bootstrap=False,
    max_depth=20,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=200,
    random_state=42
)

# Train the model 
extra_trees.fit(X_train_scaled, y_train)

# Prediction on testing set
y_pred = extra_trees.predict(X_test_scaled)

# Predicted values clipped within the extraction range [0%, 100%]
y_pred = np.clip(y_pred, 0, 1)

# Metrics for testing set
r_squared = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
aard = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Training set predictions and metrics
y_train_pred = extra_trees.predict(X_train_scaled)
y_train_pred = np.clip(y_train_pred, 0, 1)  # Clip training predictions as well
train_r_squared = r2_score(y_train, y_train_pred)
train_aard = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)

# Results
print("Optimised ExtraTreesRegressor, training and testing metrics:")
print(f"Testing - R2: {r_squared:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MSE: {mse:.4f}, AARD: {aard:.4f}%")
print(f"Training - R2: {train_r_squared:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MSE: {train_mse:.4f}, AARD: {train_aard:.4f}%")

# Plotting Experimental vs calculated E
plt.scatter(y_test, y_pred)
# Linear fit in red
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "r-", label='Best Linear Fit')

# y = x line in thinner dotted black
plt.plot(y_test, y_test, 'k-', label='y = x')

# Adjusting labels and adding legend
plt.xlabel("Experimental E")
plt.ylabel("Calculated E")
plt.legend(frameon=False)

# Save the plot to a file
plt.savefig('Experimental_calculated_E.png')
plt.close()
