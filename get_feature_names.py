import pandas as pd
from sklearn.preprocessing import StandardScaler
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
    numerical_cols = ['Year', 'Financial Loss (in Million $)', 'Number of Affected Users', 'Incident Resolution Time (in Hours)']
    scaler = StandardScaler()
    # We only need the column names, so we don't actually fit_transform here
    # Just return the dataframe with numerical columns as they are for now
    return df, scaler

if __name__ == "__main__":
    df = load_data('Global_Cybersecurity_Threats_2015-2024.csv')
    df = handle_missing_values(df)
    df = one_hot_encode(df)
    df, scaler = standardize_numerical_features(df)

    # Drop the target variable to get the feature names
    X = df.drop('Financial Loss (in Million $)', axis=1)

    # Load the RFE selector
    rfe = joblib.load('rfe.joblib')

    # Get the names of the selected features
    selected_feature_names = X.columns[rfe.support_]
    print("Selected Features:")
    for feature in selected_feature_names:
        print(feature)
