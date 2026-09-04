import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_and_preprocess_data(filepath='student_data.csv'):
    df = pd.read_csv(filepath)
    
    print("Original data shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)
    
    features = [
        'Attendance', 'Internal_Marks', 'Previous_CGPA',
        'Assignment_Score', 'Assignment_Submission', 'Study_Hours',
        'Backlogs', 'Sleep_Hours', 'Previous_Failures'
    ]
    
    X = df[features]
    y = df['Risk_Category']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)
    
    joblib.dump(scaler, 'scaler.pkl')
    print("\nScaler saved to 'scaler.pkl'")
    
    print(f"\nTraining set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    print("\nTraining set class distribution:")
    print(y_train.value_counts().sort_index())
    print("\nTest set class distribution:")
    print(y_test.value_counts().sort_index())
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess_data()