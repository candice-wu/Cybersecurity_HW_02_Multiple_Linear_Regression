import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_data(file_path):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(file_path)

if __name__ == "__main__":
    df = load_data('Global_Cybersecurity_Threats_2015-2024.csv')

    # 設定 target 與 features
    target = "Financial Loss (in Million $)"
    X = df.drop(columns=[target])
    y = df[target]

    # 類別欄位轉換
    X = pd.get_dummies(X, drop_first=True)

    # 數值標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 訓練/測試集切分
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test)

    # Save the data and scaler
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(X.columns, 'feature_names.pkl') # Save feature names for later use in Streamlit app

    # Save full processed X and y for statsmodels
    np.save('X_full_processed.npy', X)
    np.save('y_full_processed.npy', y)
