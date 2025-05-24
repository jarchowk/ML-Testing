import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
from xgboost import plot_importance

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def load_data():
    train = pd.read_csv('Car_Insurance_Claim.csv')
    return train

if __name__ == "__main__":
    print("Loading data...")
    train = load_data()
    print("Train shape:", train.shape)

    # --- 2. Data Preprocessing ---
    print("\n--- Data Preprocessing ---")

    # Identify categorical and numerical features
    categorical_features = ['AGE', 'GENDER', 'RACE', 'DRIVING_EXPERIENCE', 'EDUCATION', 'INCOME', 'VEHICLE_YEAR']
    numerical_features = ['CREDIT_SCORE', 'SPEEDING_VIOLATIONS', 'PAST_ACCIDENTS']

    # Define features and target
    feature_cols = categorical_features + numerical_features
    X = train[feature_cols]
    y = train['OUTCOME']

    # Create preprocessing pipelines for numerical and categorical features
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
        #('scaler', MinMaxScaler())
    ])

    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for dense array
    ])

    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # Split data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Validation set shape: X_val={X_val.shape}, y_val={y_val.shape}")
    print(f"Test set shape: X_test={X_test.shape}")

    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Processed training data shape: {X_train_processed.shape}")
    feature_names_out = preprocessor.get_feature_names_out()
    print(f"Number of features after preprocessing: {len(feature_names_out)}")

    neg, pos = np.bincount(y_train)
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=weight_for_1 / weight_for_0,  # handle imbalance
        use_label_encoder=False,
        eval_metric='auc'
    )

    model.fit(X_train_processed, y_train)
    y_val_pred = model.predict_proba(X_val_processed)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred)
    print(f"Validation AUC: {val_auc:.4f}")


    # Step 1: Get feature names from preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # Step 2: Map f0, f1, ..., to actual names
    xgb_feature_map = {f"f{i}": name for i, name in enumerate(feature_names)}

    # Step 3: Get feature importances from XGBoost model
    booster = model.get_booster()
    importance = booster.get_score(importance_type='gain')  # use 'gain', 'weight', etc.

    # Step 4: Map internal names to real names
    readable_importance = {
        xgb_feature_map.get(k, k): v for k, v in importance.items()
    }

    # Step 5: Create DataFrame
    importance_df = pd.DataFrame(
        list(readable_importance.items()), columns=["Feature", "Importance"]
    ).sort_values(by="Importance", ascending=False)

    # Step 6: Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"][:50][::-1], importance_df["Importance"][:50][::-1])
    plt.title("Top 50 Feature Importances (Gain)")
    plt.xlabel("Importance (Gain)")
    plt.tight_layout()
    plt.show()


    # Predict probabilities
    y_train_pred = model.predict_proba(X_train_processed)[:, 1]
    y_val_pred = model.predict_proba(X_val_processed)[:, 1]

    # Compute ROC curve and AUC for training
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
    roc_auc_train = auc(fpr_train, tpr_train)

    # Compute ROC curve and AUC for validation
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred)
    roc_auc_val = auc(fpr_val, tpr_val)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, label=f"Train ROC curve (AUC = {roc_auc_train:.2f})")
    plt.plot(fpr_val, tpr_val, label=f"Validation ROC curve (AUC = {roc_auc_val:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


print("\n--- Example Script Finished ---")