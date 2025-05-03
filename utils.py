import pandas as pd
import numpy as np
import os
import io

def load_demo_data():
    """Load the sample hybrid synthetic data from file or create it if not found"""
    # Try to load from the expected path
    try:
        data_path = "attached_assets/hybrid_synthetic_data.csv"
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        else:
            # Try alternative path
            alt_path = "hybrid_synthetic_data.csv"
            if os.path.exists(alt_path):
                return pd.read_csv(alt_path)
            else:
                # Fallback to generate synthetic data if file not found
                print("Could not find hybrid_synthetic_data.csv, creating synthetic data.")
                return create_synthetic_data()
    except Exception as e:
        print(f"Error loading demo data: {e}")
        # Return fallback synthetic data
        return create_synthetic_data()

def create_synthetic_data(n_samples=100):
    """Create synthetic data for demonstration if the real data is not available"""
    np.random.seed(42)  # For reproducibility
    
    # Create feature dataframe
    data = pd.DataFrame({
        'Age': np.random.randint(30, 90, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'BMI': np.round(np.random.uniform(20, 35, n_samples), 1),
        'Diabetes_Duration': np.random.randint(0, 30, n_samples),
        'HbA1c': np.round(np.random.uniform(5.0, 11.0, n_samples), 1),
        'Blood_Pressure_Systolic': np.random.randint(100, 180, n_samples),
        'Blood_Pressure_Diastolic': np.random.randint(50, 110, n_samples),
        'Cholesterol': np.random.randint(150, 280, n_samples),
        'Smoking_Status': np.random.choice(['Never', 'Former', 'Current'], n_samples),
        'Physical_Activity': np.random.choice(['Low', 'Moderate', 'High'], n_samples),
    })
    
    # Create boolean features with higher probability of True for older patients
    age_norm = (data['Age'] - data['Age'].min()) / (data['Age'].max() - data['Age'].min())
    
    # Higher age and higher HbA1c increase probability of complications
    risk_factor = (age_norm * 0.7) + ((data['HbA1c'] - 5.0) / 6.0) * 0.3
    
    data['Previous_Amputation'] = (np.random.random(n_samples) < risk_factor * 0.3).astype(int)
    data['Neuropathy'] = (np.random.random(n_samples) < risk_factor * 0.5).astype(int)
    data['PAD'] = (np.random.random(n_samples) < risk_factor * 0.4).astype(int)
    data['Foot_Deformity'] = (np.random.random(n_samples) < risk_factor * 0.35).astype(int)
    
    # Create target variable - higher probability with risk factors
    target_prob = (
        risk_factor * 0.4 + 
        data['Previous_Amputation'] * 0.2 + 
        data['Neuropathy'] * 0.15 + 
        data['PAD'] * 0.15 + 
        data['Foot_Deformity'] * 0.1
    )
    data['Foot_Ulcer_History'] = (np.random.random(n_samples) < target_prob).astype(int)
    
    return data

def format_metrics_table(metrics_df):
    """Format metrics dataframe for display in Streamlit"""
    # Create a copy to avoid modifying the original
    formatted_df = metrics_df.copy()
    
    # Format values as percentages with 2 decimal places except for ROC-AUC
    for col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].apply(
            lambda x: f"{x*100:.2f}%" if formatted_df.index[formatted_df[col] == x][0] != "ROC-AUC" 
            else f"{x:.3f}"
        )
    
    return formatted_df

def save_dataframe_to_csv(df):
    """Convert dataframe to csv for download"""
    buffer = io.StringIO()
    df.to_csv(buffer, index=True)
    buffer.seek(0)
    return buffer.getvalue()
