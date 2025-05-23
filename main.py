import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.set_visible_devices([], 'GPU')

def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    sample_submission = pd.read_csv('sample_submission.csv')
    return train, test, sample_submission

if __name__ == "__main__":
    print("Loading data...")
    train, test, sample_submission = load_data()
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("Sample submission shape:", sample_submission.shape)

    # --- 2. Data Preprocessing ---
    print("\n--- Data Preprocessing ---")

    # Identify categorical and numerical features
    categorical_features = ['model', 'area_cluster', 'fuel_type', 'transmission_type', 'rear_brakes_type']
    numerical_features = ['age_of_policyholder', 'policy_tenure', 'age_of_car', 'population_density']

    # Define features and target
    feature_cols = categorical_features + numerical_features
    X = train[feature_cols]
    y = train['is_claim']

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

    # Split data into training and validation sets only
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # If your test set has labels, use them; otherwise, set y_test = None
    if 'is_claim' in test.columns:
        X_test = test[feature_cols]
        y_test = test['is_claim']
    else:
        X_test = test[feature_cols]
        y_test = None

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

    # --- 3. Neural Network Model (TensorFlow/Keras) ---
    print("\n--- Building Neural Network Model ---")

    # Define the model
    # model = tf.keras.Sequential([
    #     tf.keras.layers.InputLayer(input_shape=(X_train_processed.shape[1],)),
    #     tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    #     tf.keras.layers.Dropout(0.3),
    #     tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(32, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='sigmoid') # Sigmoid for binary classification
    # ])

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train_processed.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    # For imbalanced datasets, consider class weights or other metrics like Precision, Recall, F1, AUC
    # Let's add Precision and Recall as metrics
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=METRICS)

    model.summary()

    # --- 4. Model Training ---
    print("\n--- Training Model ---")

    # Add a callback for early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', # Monitor validation AUC
        patience=15,       # Number of epochs with no improvement after which training will be stopped
        mode='max',        # For AUC, we want to maximize it
        restore_best_weights=True # Restores model weights from the epoch with the best value of the monitored quantity.
    )

    # Calculate class weights for imbalanced data (optional, but good practice)
    neg, pos = np.bincount(y_train)
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"Class weights: {class_weight}")

    history = model.fit(
        X_train_processed,
        y_train,
        epochs=100, # Max epochs
        batch_size=32,
        validation_data=(X_val_processed, y_val),
        callbacks=[early_stopping],
        class_weight=class_weight, # Use class weights
        verbose=1
    )

    # --- 5. Model Evaluation ---
    print("\n--- Evaluating Model ---")

    # Evaluate on the test set

    if y_test is not None:
        print("Evaluating on Test Set:")
        results = model.evaluate(X_test_processed, y_test, batch_size=128, verbose=0)
        for name, value in zip(model.metrics_names, results):
            print(f"{name}: {value:.4f}")

        # Predictions
        y_pred_proba = model.predict(X_test_processed).ravel()
        y_pred_classes = (y_pred_proba > 0.5).astype(int)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=['No Claim (0)', 'Claim (1)']))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Claim', 'Claim'], yticklabels=['No Claim', 'Claim'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    else:
        print("No test labels available. Skipping evaluation on test set.")

    # ROC AUC Score (already part of model.evaluate if AUC metric is used)
    # roc_auc = roc_auc_score(y_test, y_pred_proba)
    # print(f"\nROC AUC Score: {roc_auc:.4f}") # This should be close to the 'auc' from model.evaluate

    # Plot training history
    def plot_training_history(history):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        # Plot accuracy
        ax[0].plot(history.history['accuracy'], label='Train Accuracy')
        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax[0].set_title('Model Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend()

        # Plot loss
        ax[1].plot(history.history['loss'], label='Train Loss')
        ax[1].plot(history.history['val_loss'], label='Validation Loss')
        ax[1].set_title('Model Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].legend()

        plt.tight_layout()
        plt.show()

        # Plot other metrics if available
        if 'auc' in history.history:
            plt.figure(figsize=(7,5))
            plt.plot(history.history['auc'], label='Train AUC')
            plt.plot(history.history['val_auc'], label='Validation AUC')
            plt.title('Model AUC')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.legend()
            plt.show()

    plot_training_history(history)

    # --- (Optional) Save and Load Model ---
    # print("\n--- Saving Model ---")
    # model.save('car_insurance_claim_model.keras') # Recommended Keras format
    # print("Model saved as car_insurance_claim_model.keras")

    # To load the model later:
    # loaded_model = tf.keras.models.load_model('car_insurance_claim_model.keras')
    # print("Model loaded. Making a test prediction:")
    # sample_prediction = loaded_model.predict(X_test_processed[:1])
    # print(f"Prediction for first test sample: {sample_prediction}")

print("\n--- Example Script Finished ---")