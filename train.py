import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import sys

def main():
    print("Starting model training process...")

    # 1. Dataset Loading and Validation
    file_path = "US accident datset.csv"
    if not os.path.exists(file_path):
        print(f"Error: Dataset {file_path} not found.")
        sys.exit(1)
        
    print(f"Loading dataset: {file_path}")
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        # Fix typo in dataset header
        df.rename(columns={'oad_Type': 'Road_Type'}, inplace=True)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)
    
    print(f"Initial Dataset Shape: {df.shape}")

    # 3. Feature Processing (defined in requirements)
    feature_cols = [
        'Road_Type', 'Road_Condition', 'Weather_Condition', 'Vehicle_Speed', 
        'Speed_Limit', 'Vehicle_Type', 'Temperature', 'Humidity', 'Visibility', 
        'T_Junction', 'Crossing', 'Railway_Crossing', 'Stop_Signal', 'Speed_Breaker'
    ]

    # Verify all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in dataset: {missing_cols}")
        sys.exit(1)
        
    if 'Severity_Level' not in df.columns:
        print("Error: Target column 'Severity_Level' missing.")
        sys.exit(1)

    # Handle missing values safely (drop rows where critical features are NA)
    df.dropna(subset=feature_cols + ['Severity_Level'], inplace=True)
    print(f"Dataset Shape after dropping missing values: {df.shape}")

    # 2. Severity Levels processing
    # Remove 'Critical' and any invalid severity rows, keeping only Low, Medium, High
    valid_severities = ['Low', 'Medium', 'High']
    df = df[df['Severity_Level'].isin(valid_severities)].copy()
    
    if len(df) == 0:
        print("Error: No valid severity rows found in dataset.")
        sys.exit(1)
    
    print("Mapping Severity labels to numeric (1=Low, 2=Medium, 3=High)...")
    severity_mapper = {'Low': 1, 'Medium': 2, 'High': 3}
    df['Severity_Level'] = df['Severity_Level'].map(severity_mapper)
    
    print(f"Severity distribution:\n{df['Severity_Level'].value_counts().sort_index()}")

    X = df[feature_cols].copy()
    y = df['Severity_Level']

    # 4. Encoding
    encoders = {}
    categorical_cols = ['Road_Type', 'Road_Condition', 'Weather_Condition', 'Vehicle_Type']

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        print(f"Encoded '{col}': {list(le.classes_)}")

    print(f"\nFinal feature set ({len(feature_cols)} features): {feature_cols}")

    # 5. Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining Multiple Models for Comparison...")
    # 6. XGBoost Configuration for 3 classes
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "XGBoost": xgb.XGBClassifier(
            objective='multi:softmax', 
            num_class=3, 
            eval_metric='mlogloss', 
            random_state=42
        )
    }

    best_model = None
    best_acc = 0
    best_pred = None
    best_name = ""

    for name, clf in models.items():
        try:
            if name == "XGBoost":
                # XGBoost expects 0-indexed classes (0, 1, 2)
                clf.fit(X_train, y_train - 1)
                preds = clf.predict(X_test) + 1
            else:
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                
            acc = accuracy_score(y_test, preds)
            print(f"  -> {name} Accuracy: {acc * 100:.2f}%")
            
            if acc > best_acc:
                best_acc = acc
                best_model = clf
                best_pred = preds
                best_name = name
        except Exception as e:
            print(f"Error training {name}: {e}")

    if not best_model:
        print("Error: No models were successfully trained.")
        sys.exit(1)

    # 7. Output
    print("\n" + "=" * 50)
    print(f"Best Model Selected: {best_name} with {best_acc * 100:.2f}% Accuracy")
    print(f"Total valid records used: {len(df)}")
    print("\nClassification Report (Best Model):")
    # We define it manually for safety
    target_names = ['Low (1)', 'Medium (2)', 'High (3)']
    print(classification_report(y_test, best_pred, target_names=target_names, zero_division=0))
    print("=" * 50)

    # 8. Save Files
    os.makedirs('model', exist_ok=True)
    
    # Save the required artifacts
    joblib.dump(best_model, 'model/model.pkl')
    joblib.dump(feature_cols, 'model/features.pkl')
    joblib.dump(encoders, 'model/encoders.pkl')
    
    severity_map = {1: "Low", 2: "Medium", 3: "High"}
    joblib.dump(severity_map, 'model/severity_map.pkl')

    print("\nModel artifacts saved to 'model/' directory:")
    print(" - model.pkl")
    print(" - features.pkl")
    print(" - encoders.pkl")
    print(" - severity_map.pkl")
    print("Training complete!")

if __name__ == '__main__':
    main()
