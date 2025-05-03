import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier  # Replace TensorFlow with sklearn MLP
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from scipy.special import expit
import os
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def prepare_test_data(data, test_size=0.2):
    """Prepare data for testing by preprocessing and splitting it"""
    # Encode categorical features
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le
    
    # Split features and target
    X = data.drop(columns=['Foot_Ulcer_History'])
    y = data['Foot_Ulcer_History']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

def run_model(data):
    """Run the HDFUPM model and return all necessary results for the dashboard"""
    
    # Preprocess and split data
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_test_data(data)
    
    # ----------------------------------------
    # Model Training
    # ----------------------------------------
    
    # Train Random Forest (RF)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # RF Probabilities
    rf_probs_train = rf_model.predict_proba(X_train_scaled)[:, 1]
    rf_probs_test = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # Train DNN Model using sklearn MLPClassifier instead of TensorFlow
    dnn_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )
    
    # Train the MLP model
    dnn_model.fit(X_train_scaled, y_train)
    
    # DNN Probabilities
    dnn_probs_train = dnn_model.predict_proba(X_train_scaled)[:, 1]  # Get probability of positive class
    dnn_probs_test = dnn_model.predict_proba(X_test_scaled)[:, 1]
    
    # ----------------------------------------
    # Adaptive Fusion Mechanism
    # ----------------------------------------
    
    # Adaptive weight calculation using logit
    logit_rf_train = np.log(np.clip(rf_probs_train, 1e-10, 1-1e-10) / np.clip(1 - rf_probs_train, 1e-10, 1-1e-10))
    logit_dnn_train = np.log(np.clip(dnn_probs_train, 1e-10, 1-1e-10) / np.clip(1 - dnn_probs_train, 1e-10, 1-1e-10))
    
    # Calculate adaptive weights
    alpha_rf = np.mean(logit_rf_train) / (np.mean(logit_rf_train) + np.mean(logit_dnn_train))
    alpha_dnn = 1 - alpha_rf
    
    # Combined Logits for Train and Test
    train_combined = alpha_rf * logit_rf_train + alpha_dnn * logit_dnn_train
    
    # Avoid division by zero in logits for test data
    rf_probs_test_safe = np.clip(rf_probs_test, 1e-10, 1-1e-10)
    dnn_probs_test_safe = np.clip(dnn_probs_test, 1e-10, 1-1e-10)
    
    test_combined = alpha_rf * np.log(rf_probs_test_safe / (1 - rf_probs_test_safe)) + alpha_dnn * np.log(dnn_probs_test_safe / (1 - dnn_probs_test_safe))
    
    # Sigmoid to get final probabilities
    test_final_probs = expit(test_combined)
    
    # ----------------------------------------
    # Risk Percentage and Category
    # ----------------------------------------
    
    # Calculate risk percentage
    risk_percentage = test_final_probs * 100
    
    # Define risk category
    def categorize_risk(prob):
        if prob < 30:
            return "Low Risk"
        elif 30 <= prob <= 70:
            return "Medium Risk"
        else:
            return "High Risk"
    
    # Apply risk category to each patient
    risk_category = [categorize_risk(prob) for prob in risk_percentage]
    
    # Create risk results dataframe
    risk_results_df = pd.DataFrame({
        "Patient_ID": range(1, len(risk_percentage) + 1),
        "Risk_Percentage": [round(x, 2) for x in risk_percentage],
        "Risk_Category": risk_category
    })
    
    # ----------------------------------------
    # Survival Analysis
    # ----------------------------------------
    
    # Create survival data based on risk percentage (higher risk = lower survival)
    survival_data = []
    for i, risk in enumerate(risk_percentage):
        # Calculate survival probabilities inversely related to risk
        year1_surv = max(0.05, min(0.95, 1 - (risk/100) * 0.8))  # Scale to avoid 0 or 1
        year3_surv = max(0.05, min(0.95, year1_surv * (0.9 - risk/500)))  # Decrease over time
        year5_surv = max(0.05, min(0.95, year3_surv * (0.8 - risk/500)))  # Further decrease
        
        survival_data.append({
            "Patient_ID": i + 1,
            "1-Year Survival": round(year1_surv, 2),
            "3-Year Survival": round(year3_surv, 2),
            "5-Year Survival": round(year5_surv, 2)
        })
    
    survival_results_df = pd.DataFrame(survival_data)
    
    # ----------------------------------------
    # Model Evaluation
    # ----------------------------------------
    
    # Convert final probabilities to binary predictions
    final_preds_binary = (test_final_probs >= 0.5).astype(int)
    rf_preds_binary = (rf_probs_test >= 0.5).astype(int)
    dnn_preds_binary = (dnn_probs_test >= 0.5).astype(int)
    
    # Calculate metrics for hybrid model
    hybrid_accuracy = accuracy_score(y_test, final_preds_binary)
    hybrid_precision = precision_score(y_test, final_preds_binary, zero_division=0)
    hybrid_recall = recall_score(y_test, final_preds_binary, zero_division=0)
    hybrid_f1 = f1_score(y_test, final_preds_binary, zero_division=0)
    hybrid_roc_auc = roc_auc_score(y_test, test_final_probs) if len(np.unique(y_test)) > 1 else 0.5
    
    # Calculate metrics for Random Forest
    rf_accuracy = accuracy_score(y_test, rf_preds_binary)
    rf_precision = precision_score(y_test, rf_preds_binary, zero_division=0)
    rf_recall = recall_score(y_test, rf_preds_binary, zero_division=0)
    rf_f1 = f1_score(y_test, rf_preds_binary, zero_division=0)
    rf_roc_auc = roc_auc_score(y_test, rf_probs_test) if len(np.unique(y_test)) > 1 else 0.5
    
    # Calculate metrics for DNN
    dnn_accuracy = accuracy_score(y_test, dnn_preds_binary)
    dnn_precision = precision_score(y_test, dnn_preds_binary, zero_division=0)
    dnn_recall = recall_score(y_test, dnn_preds_binary, zero_division=0)
    dnn_f1 = f1_score(y_test, dnn_preds_binary, zero_division=0)
    dnn_roc_auc = roc_auc_score(y_test, dnn_probs_test) if len(np.unique(y_test)) > 1 else 0.5
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(
        {
            "Random Forest": [rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc],
            "Deep Neural Network": [dnn_accuracy, dnn_precision, dnn_recall, dnn_f1, dnn_roc_auc],
            "Hybrid Model": [hybrid_accuracy, hybrid_precision, hybrid_recall, hybrid_f1, hybrid_roc_auc]
        },
        index=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    )
    
    # Calculate ROC curve for hybrid model
    fpr, tpr, _ = roc_curve(y_test, test_final_probs)
    roc_data = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    
    # Return all results
    return {
        "metrics": metrics_df,
        "risk_results": risk_results_df,
        "survival_results": survival_results_df,
        "roc_data": roc_data
    }
