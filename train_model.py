import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import joblib
import statsmodels.api as sm
import pandas as pd

# Load the data for sklearn model
X_train = np.load('X_train.npy', allow_pickle=True)
y_train = np.load('y_train.npy', allow_pickle=True)

# Create a Linear Regression model
model = LinearRegression()

# Use Recursive Feature Elimination (RFE) to select the best features
# In the previous notebook, we found that 10 features were optimal.
rfe = RFE(model, n_features_to_select=10)
X_train_rfe = rfe.fit_transform(X_train, y_train)

# Train the sklearn model on the selected features
model.fit(X_train_rfe, y_train)

# Save the trained sklearn model and the RFE selector
joblib.dump(model, 'cyber_risk_model.pkl')
joblib.dump(rfe, 'rfe.joblib')

print("Sklearn model trained and saved successfully.")

# --- Fit and Save Statsmodels OLS Model ---

# Load full processed X and y for statsmodels
X_full_processed = np.load('X_full_processed.npy', allow_pickle=True)
y_full_processed = np.load('y_full_processed.npy', allow_pickle=True)
feature_names = joblib.load('feature_names.pkl')

# Create DataFrame for statsmodels and ensure numeric types
X_sm = pd.DataFrame(X_full_processed, columns=feature_names).astype(float)
y_sm = pd.Series(y_full_processed).astype(float)

# Add a constant to the X for statsmodels (for intercept)
X_sm = sm.add_constant(X_sm)

# Get the names of the features selected by RFE
# Ensure 'const' is included if it was added to X_sm
selected_feature_names_rfe = feature_names[rfe.support_].tolist()
if 'const' not in selected_feature_names_rfe:
    selected_feature_names_rfe.insert(0, 'const')

# Filter X_sm to include only the RFE-selected features
X_sm_selected = X_sm[selected_feature_names_rfe]

# Fit statsmodels OLS model
sm_model = sm.OLS(y_sm, X_sm_selected).fit()

# Save the statsmodels model
joblib.dump(sm_model, 'statsmodels_model.pkl')

print("Statsmodels OLS model trained and saved successfully.")
