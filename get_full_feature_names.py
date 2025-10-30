import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_data(file_path):
    return pd.read_csv(file_path)

def handle_missing_values(df):
    return df

def one_hot_encode(df):
    categorical_cols = ['Country', 'Attack Type', 'Target Industry', 'Attack Source', 'Security Vulnerability Type', 'Defense Mechanism Used']
    df = pd.get_dummies(df, columns=categorical_cols)
    return df

def standardize_numerical_features(df):
    numerical_cols = ['Year', 'Number of Affected Users', 'Incident Resolution Time (in Hours)']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler

if __name__ == "__main__":
    df = load_data('Global_Cybersecurity_Threats_2015-2024.csv')
    df = handle_missing_values(df)
    df = one_hot_encode(df)

    # Separate features (X) and target (y)
    X_full = df.drop('Financial Loss (in Million $)', axis=1)

    print("Full Feature Names (in order):")
    for feature in X_full.columns:
        print(feature)
