
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
X_test = np.load('X_test.npy', allow_pickle=True)
y_test = np.load('y_test.npy', allow_pickle=True)

# Load the trained model and RFE selector
model = joblib.load('cyber_risk_model.pkl')
rfe = joblib.load('rfe.joblib')

# Make predictions
y_pred = model.predict(X_test[:, rfe.support_])

# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"R²: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")

# Visualize the results
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Financial Loss")
plt.savefig('actual_vs_predicted.png')
print("Actual vs. Predicted plot saved as actual_vs_predicted.png")
