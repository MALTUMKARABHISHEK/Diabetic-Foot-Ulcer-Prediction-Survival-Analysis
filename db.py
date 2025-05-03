"""
Database utilities for storing and retrieving diabetic foot ulcer risk prediction data.
"""
import os
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
import json

# Database connection
def get_db_connection():
    """Connect to the PostgreSQL database server using environment variables"""
    conn = psycopg2.connect(
        os.environ.get('DATABASE_URL') or 
        "dbname={} user={} password={} host={} port={}".format(
            os.environ.get('PGDATABASE', 'postgres'),
            os.environ.get('PGUSER', 'postgres'),
            os.environ.get('PGPASSWORD', 'postgres'),
            os.environ.get('PGHOST', 'localhost'),
            os.environ.get('PGPORT', '5432')
        )
    )
    return conn

def initialize_database():
    """Create necessary tables if they don't exist"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Risk Assessment Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS risk_assessments (
                    id SERIAL PRIMARY KEY,
                    patient_id INTEGER NOT NULL,
                    risk_percentage FLOAT NOT NULL,
                    risk_category VARCHAR(20) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Survival Analysis Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS survival_analysis (
                    id SERIAL PRIMARY KEY,
                    patient_id INTEGER NOT NULL,
                    one_year_survival FLOAT NOT NULL,
                    three_year_survival FLOAT NOT NULL,
                    five_year_survival FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Model Performance Metrics Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(50) NOT NULL,
                    accuracy FLOAT NOT NULL,
                    precision FLOAT NOT NULL,
                    recall FLOAT NOT NULL,
                    f1_score FLOAT NOT NULL,
                    roc_auc FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # ROC Curve Data Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS roc_data (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(50) NOT NULL,
                    fpr JSONB NOT NULL,
                    tpr JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            conn.commit()
            print("Database tables created successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        conn.close()

def store_risk_assessments(risk_results_df):
    """Store risk assessment results in the database"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for _, row in risk_results_df.iterrows():
                cur.execute("""
                    INSERT INTO risk_assessments (patient_id, risk_percentage, risk_category)
                    VALUES (%s, %s, %s)
                    RETURNING id;
                """, (
                    int(row['Patient_ID']),
                    float(row['Risk_Percentage']),
                    row['Risk_Category']
                ))
            
            conn.commit()
            return True
    except Exception as e:
        conn.rollback()
        print(f"Error storing risk assessments: {e}")
        return False
    finally:
        conn.close()

def store_survival_analysis(survival_results_df):
    """Store survival analysis results in the database"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for _, row in survival_results_df.iterrows():
                cur.execute("""
                    INSERT INTO survival_analysis (
                        patient_id, one_year_survival, three_year_survival, five_year_survival
                    )
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                """, (
                    int(row['Patient_ID']),
                    float(row['1-Year Survival']),
                    float(row['3-Year Survival']),
                    float(row['5-Year Survival'])
                ))
            
            conn.commit()
            return True
    except Exception as e:
        conn.rollback()
        print(f"Error storing survival analysis: {e}")
        return False
    finally:
        conn.close()

def store_model_metrics(metrics_df):
    """Store model performance metrics in the database"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for model_name in metrics_df.columns:
                model_metrics = metrics_df[model_name]
                
                cur.execute("""
                    INSERT INTO model_metrics (
                        model_name, accuracy, precision, recall, f1_score, roc_auc
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id;
                """, (
                    model_name,
                    float(model_metrics['Accuracy']),
                    float(model_metrics['Precision']),
                    float(model_metrics['Recall']),
                    float(model_metrics['F1-Score']),
                    float(model_metrics['ROC-AUC'])
                ))
            
            conn.commit()
            return True
    except Exception as e:
        conn.rollback()
        print(f"Error storing model metrics: {e}")
        return False
    finally:
        conn.close()

def store_roc_data(roc_data, model_name="Hybrid Model"):
    """Store ROC curve data in the database"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO roc_data (model_name, fpr, tpr)
                VALUES (%s, %s, %s)
                RETURNING id;
            """, (
                model_name,
                json.dumps(roc_data['fpr'].tolist()),
                json.dumps(roc_data['tpr'].tolist())
            ))
            
            conn.commit()
            return True
    except Exception as e:
        conn.rollback()
        print(f"Error storing ROC data: {e}")
        return False
    finally:
        conn.close()

def get_latest_risk_assessments():
    """Retrieve the latest risk assessments from the database"""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT patient_id as "Patient_ID", 
                       risk_percentage as "Risk_Percentage", 
                       risk_category as "Risk_Category"
                FROM risk_assessments
                ORDER BY created_at DESC
                LIMIT 100;
            """)
            results = cur.fetchall()
            return pd.DataFrame(results)
    except Exception as e:
        print(f"Error retrieving risk assessments: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_latest_survival_analysis():
    """Retrieve the latest survival analysis from the database"""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT patient_id as "Patient_ID", 
                       one_year_survival as "1-Year Survival", 
                       three_year_survival as "3-Year Survival", 
                       five_year_survival as "5-Year Survival"
                FROM survival_analysis
                ORDER BY created_at DESC
                LIMIT 100;
            """)
            results = cur.fetchall()
            return pd.DataFrame(results)
    except Exception as e:
        print(f"Error retrieving survival analysis: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_latest_model_metrics():
    """Retrieve the latest model performance metrics from the database"""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                WITH latest_metrics AS (
                    SELECT model_name, MAX(created_at) as max_created_at
                    FROM model_metrics
                    GROUP BY model_name
                )
                SELECT mm.model_name, mm.accuracy, mm.precision, mm.recall, mm.f1_score, mm.roc_auc
                FROM model_metrics mm
                JOIN latest_metrics lm ON mm.model_name = lm.model_name AND mm.created_at = lm.max_created_at;
            """)
            results = cur.fetchall()
            
            # Convert to DataFrame with model_name as columns
            if results:
                metrics_dict = {
                    metric: [] for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
                }
                
                model_names = []
                for row in results:
                    model_name = row['model_name']
                    model_names.append(model_name)
                    
                    metrics_dict["Accuracy"].append(row['accuracy'])
                    metrics_dict["Precision"].append(row['precision'])
                    metrics_dict["Recall"].append(row['recall'])
                    metrics_dict["F1-Score"].append(row['f1_score'])
                    metrics_dict["ROC-AUC"].append(row['roc_auc'])
                
                df = pd.DataFrame(metrics_dict, index=model_names).T
                return df
            
            return pd.DataFrame()
    except Exception as e:
        print(f"Error retrieving model metrics: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_latest_roc_data(model_name="Hybrid Model"):
    """Retrieve the latest ROC curve data from the database"""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT fpr, tpr 
                FROM roc_data
                WHERE model_name = %s
                ORDER BY created_at DESC
                LIMIT 1;
            """, (model_name,))
            result = cur.fetchone()
            
            if result:
                return pd.DataFrame({
                    'fpr': json.loads(result['fpr']),
                    'tpr': json.loads(result['tpr'])
                })
            
            return pd.DataFrame()
    except Exception as e:
        print(f"Error retrieving ROC data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def store_all_results(model_results):
    """Store all model results to the database"""
    try:
        # Initialize the database if needed
        initialize_database()
        
        # Store risk assessments
        store_risk_assessments(model_results['risk_results'])
        
        # Store survival analysis
        store_survival_analysis(model_results['survival_results'])
        
        # Store model metrics
        store_model_metrics(model_results['metrics'])
        
        # Store ROC data
        store_roc_data(model_results['roc_data'])
        
        return True
    except Exception as e:
        print(f"Error storing all results: {e}")
        return False