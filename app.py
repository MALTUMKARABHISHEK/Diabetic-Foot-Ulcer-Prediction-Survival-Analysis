import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
from model import run_model, prepare_test_data
from utils import load_demo_data, save_dataframe_to_csv, format_metrics_table

# Page configuration
st.set_page_config(
    page_title="Diabetic Foot Ulcer Risk Prediction Dashboard",
    page_icon="🩺",
    layout="wide"
)

# Application title
st.title("Diabetic Foot Ulcer Risk Prediction Dashboard")

# Sidebar for navigation and controls
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "Model Performance", "Risk Assessment", "Survival Analysis", "ROC Curve Analysis"]
)

# Initialize session state for storing model results
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

# Data upload section
st.sidebar.title("Data Input")
data_option = st.sidebar.radio(
    "Choose data source:",
    ["Use Demo Data", "Upload Your Own Data"]
)

if data_option == "Upload Your Own Data":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = data
        except Exception as e:
            st.sidebar.error(f"Error loading data: {e}")
            st.session_state.uploaded_data = None
else:
    # Use demo data
    if st.sidebar.button("Load Demo Data"):
        try:
            data = load_demo_data()
            st.session_state.uploaded_data = data
            st.sidebar.success("Demo data loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading demo data: {e}")
            st.session_state.uploaded_data = None

# Run model button
if st.session_state.uploaded_data is not None:
    if st.sidebar.button("Run Model"):
        with st.spinner("Running model predictions..."):
            # Run the model
            st.session_state.model_results = run_model(st.session_state.uploaded_data)
            st.sidebar.success("Model execution completed!")

# HOME PAGE
if page == "Home":
    st.markdown("""
    ## Welcome to the Diabetic Foot Ulcer Risk Prediction Dashboard
    
    This dashboard allows you to visualize and analyze risk predictions for diabetic foot ulcers using various machine learning models.
    
    ### Key Features:
    
    1. **Model Performance Comparison**: Compare metrics between Random Forest, Deep Neural Network, and Hybrid models
    2. **Risk Assessment**: View patient-specific risk percentages and categories
    3. **Survival Analysis**: Examine survival probabilities at 1, 3, and 5 years
    4. **ROC Curve Analysis**: Visualize the ROC curve for the hybrid model
    
    ### Getting Started:
    
    1. Use the sidebar to navigate between different analysis pages
    2. Choose between demo data or upload your own CSV file
    3. Run the model to generate predictions
    4. Explore the visualizations and download reports as needed
    
    ### Required Data Format:
    
    Your CSV should include the following columns:
    - Age, Gender, BMI, Diabetes_Duration, HbA1c
    - Blood pressure readings (Systolic and Diastolic)
    - Risk factors like Smoking_Status, Physical_Activity
    - Medical history indicators like Previous_Amputation, Neuropathy, PAD
    - Target variable: Foot_Ulcer_History
    """)
    
    # Display sample data if available
    if st.session_state.uploaded_data is not None:
        st.subheader("Preview of the Current Dataset")
        st.dataframe(st.session_state.uploaded_data.head())

# MODEL PERFORMANCE PAGE
elif page == "Model Performance":
    st.header("Model Performance Comparison")
    
    if st.session_state.model_results is None:
        st.info("Please upload data and run the model to view performance metrics.")
    else:
        metrics = st.session_state.model_results['metrics']
        
        # Create and display metrics table
        st.subheader("Performance Metrics")
        metric_table = format_metrics_table(metrics)
        st.table(metric_table)
        
        # Add download button for metrics
        metrics_csv = save_dataframe_to_csv(metric_table)
        st.download_button(
            label="Download Metrics as CSV",
            data=metrics_csv,
            file_name="model_performance_metrics.csv",
            mime="text/csv"
        )
        
        # Bar chart for metrics comparison
        st.subheader("Visual Comparison")
        
        # Create Plotly figure for model comparison
        metrics_for_plot = metrics.copy()
        model_names = ["Random Forest", "Deep Neural Network", "Hybrid Model"]
        
        fig = make_subplots(rows=1, cols=5, subplot_titles=("Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"))
        
        for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]):
            for j, model in enumerate(model_names):
                fig.add_trace(
                    go.Bar(
                        x=[model], 
                        y=[metrics_for_plot.loc[metric, model]], 
                        name=f"{model} - {metric}",
                        showlegend=False
                    ),
                    row=1, col=i+1
                )
        
        fig.update_layout(height=400, width=1000)
        st.plotly_chart(fig)

# RISK ASSESSMENT PAGE
elif page == "Risk Assessment":
    st.header("Risk Assessment for Foot Ulcer")
    
    if st.session_state.model_results is None:
        st.info("Please upload data and run the model to view risk assessments.")
    else:
        risk_results = st.session_state.model_results['risk_results']
        
        # Display risk results table
        st.subheader("Patient Risk Assessment")
        st.dataframe(risk_results)
        
        # Add download button for risk assessment
        risk_csv = save_dataframe_to_csv(risk_results)
        st.download_button(
            label="Download Risk Assessment as CSV",
            data=risk_csv,
            file_name="risk_assessment.csv",
            mime="text/csv"
        )
        
        # Risk distribution visualization
        st.subheader("Risk Category Distribution")
        
        # Count of patients in each risk category
        risk_category_counts = risk_results['Risk_Category'].value_counts().reset_index()
        risk_category_counts.columns = ['Risk Category', 'Count']
        
        # Create donut chart
        fig = px.pie(
            risk_category_counts, 
            values='Count', 
            names='Risk Category',
            title='Distribution of Risk Categories',
            hole=0.4,
            color='Risk Category',
            color_discrete_map={'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig)
        
        # Individual patient risk exploration
        st.subheader("Explore Individual Patient Risk")
        
        # Patient selector
        patient_ids = risk_results['Patient_ID'].tolist()
        selected_patient = st.selectbox("Select a patient to view detailed risk information:", patient_ids)
        
        # Display selected patient data
        if selected_patient:
            patient_data = risk_results[risk_results['Patient_ID'] == selected_patient]
            
            # Create three columns for key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Patient ID", selected_patient)
            
            with col2:
                risk_percentage = float(patient_data['Risk_Percentage'].values[0])
                st.metric("Risk Percentage", f"{risk_percentage:.2f}%")
            
            with col3:
                risk_category = patient_data['Risk_Category'].values[0]
                st.metric("Risk Category", risk_category)

# SURVIVAL ANALYSIS PAGE
elif page == "Survival Analysis":
    st.header("Survival Analysis")
    
    if st.session_state.model_results is None:
        st.info("Please upload data and run the model to view survival analysis.")
    else:
        survival_results = st.session_state.model_results['survival_results']
        
        # Display survival results table
        st.subheader("Patient Survival Probabilities")
        st.dataframe(survival_results)
        
        # Add download button for survival analysis
        survival_csv = save_dataframe_to_csv(survival_results)
        st.download_button(
            label="Download Survival Analysis as CSV",
            data=survival_csv,
            file_name="survival_analysis.csv",
            mime="text/csv"
        )
        
        # Visualization: Survival probability comparison
        st.subheader("Survival Probability Comparison")
        
        # Patient selector for detailed view
        patient_ids = survival_results['Patient_ID'].tolist()
        selected_patients = st.multiselect(
            "Select patients to compare survival probabilities:", 
            patient_ids,
            default=patient_ids[:5] if len(patient_ids) >= 5 else patient_ids
        )
        
        if selected_patients:
            filtered_data = survival_results[survival_results['Patient_ID'].isin(selected_patients)]
            
            # Prepare data for line chart
            patients = []
            for _, row in filtered_data.iterrows():
                patient_id = row['Patient_ID']
                for year, col in [(1, '1-Year Survival'), (3, '3-Year Survival'), (5, '5-Year Survival')]:
                    patients.append({
                        'Patient ID': f"Patient {patient_id}",
                        'Year': year,
                        'Survival Probability': row[col]
                    })
            
            df_for_chart = pd.DataFrame(patients)
            
            # Create the line chart
            fig = px.line(
                df_for_chart, 
                x='Year', 
                y='Survival Probability', 
                color='Patient ID',
                markers=True,
                title='Survival Probability Over Time',
                labels={'Survival Probability': 'Probability of No Foot Ulcer'}
            )
            
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Survival Probability",
                yaxis=dict(range=[0, 1])
            )
            
            # Add reference line at 0.5
            fig.add_shape(
                type="line",
                x0=0,
                y0=0.5,
                x1=6,
                y1=0.5,
                line=dict(color="grey", width=1, dash="dash")
            )
            
            st.plotly_chart(fig)

# ROC CURVE ANALYSIS PAGE
elif page == "ROC Curve Analysis":
    st.header("ROC Curve Analysis")
    
    if st.session_state.model_results is None:
        st.info("Please upload data and run the model to view ROC curves.")
    else:
        # Get ROC curve data
        roc_data = st.session_state.model_results['roc_data']
        metrics = st.session_state.model_results['metrics']
        
        # Hybrid model ROC-AUC value
        hybrid_auc = metrics.loc['ROC-AUC', 'Hybrid Model']
        
        # Create the ROC curve figure
        fig = px.line(
            roc_data, 
            x='fpr', 
            y='tpr', 
            title=f'Receiver Operating Characteristic (ROC) Curve - Hybrid Model (AUC = {hybrid_auc:.3f})',
            labels={'fpr': 'False Positive Rate', 'tpr': 'True Positive Rate'}
        )
        
        # Add the diagonal reference line (random classifier)
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=1, y1=1,
            line=dict(color='grey', dash='dash')
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=700,
            height=500
        )
        
        # Display the plot
        st.plotly_chart(fig)
        
        # ROC Curve information
        st.subheader("Understanding the ROC Curve")
        st.markdown("""
        The Receiver Operating Characteristic (ROC) curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.
        
        **Key points:**
        - The closer the curve follows the top-left corner, the better the model's performance
        - The area under the curve (AUC) ranges from 0 to 1, with higher values indicating better performance
        - A random classifier would give a point along the diagonal line (AUC = 0.5)
        """)
        
        
