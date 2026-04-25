"""
Behavioral Analytics for Quantifying Student Self-Discipline
Based on the CAPIR Framework (Connectivity, Acquisition, Productivity, Interactivity, Reactivity)
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

def generate_synthetic_lms_data(num_students=1000):
    """
    Generates synthetic LMS interaction data since raw institutional logs are private.
    """
    np.random.seed(42)
    
    data = {
        'student_id': range(1, num_students + 1),
        # Connectivity metrics
        'f_cons': np.random.poisson(lam=15, size=num_students), # Visit frequency
        'd_session': np.random.normal(loc=45, scale=15, size=num_students), # Mean session duration
        'n_units': np.random.poisson(lam=20, size=num_students), # Unique content units accessed
        # Acquisition metrics
        's_quiz': np.random.uniform(low=40, high=100, size=num_students), # Mean formative score
        'n_achv': np.random.poisson(lam=10, size=num_students), # Completed learning objectives
        # Productivity metrics
        'g_work': np.random.uniform(low=50, high=100, size=num_students), # Workshop grades
        'g_peer': np.random.uniform(low=50, high=100, size=num_students), # Peer-review grades
        'g_assig': np.random.uniform(low=50, high=100, size=num_students), # Independent assignment grades
        # Interactivity metrics
        'n_post': np.random.poisson(lam=5, size=num_students), # Original forum posts
        'n_forum': np.random.poisson(lam=15, size=num_students), # Peer posts consulted
        # Reactivity metrics
        'r_delay': np.random.exponential(scale=24, size=num_students), # Latency in hours
        # Target variable: End of semester risk status (1 = At-Risk, 0 = Safe)
        'is_at_risk': np.random.choice([0, 1], p=[0.85, 0.15], size=num_students) 
    }
    return pd.DataFrame(data)

def calculate_capir_features(df):
    """
    Applies the CAPIR mathematical formulation to raw metrics.
    """
    # Connectivity (C): C = 0.1*f_cons + 0.3*d_session + 0.6*n_units
    df['C'] = 0.1 * df['f_cons'] + 0.3 * df['d_session'] + 0.6 * df['n_units']
    
    # Acquisition (A): A = 0.7*s_quiz + 0.3*n_achv
    df['A'] = 0.7 * df['s_quiz'] + 0.3 * df['n_achv']
    
    # Productivity (P): P = 0.4*g_work + 0.2*g_peer + 0.4*g_assig
    df['P'] = 0.4 * df['g_work'] + 0.2 * df['g_peer'] + 0.4 * df['g_assig']
    
    # Interactivity (I): I = 0.8*n_post + 0.2*n_forum
    df['I'] = 0.8 * df['n_post'] + 0.2 * df['n_forum']
    
    # Reactivity (R): Extracted directly as latency (lower is better)
    df['R'] = df['r_delay']
    
    return df

def perform_clustering(features_df):
    """
    Segments student populations into behaviorally coherent clusters (K=4).
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    
    # K-Means with optimal K=4 as per the study
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    return cluster_labels

def train_predictive_model(X, y):
    """
    Trains an SVM classifier with RBF kernel and applies SMOTE for at-risk detection.
    """
    # Train-test split (simulating early-semester predictions)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Apply SMOTE to handle class imbalance (minority failing class)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # SVM Classifier with Radial Basis Function kernel
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train_resampled)
    
    # Predictions and Evaluation
    y_pred = svm_model.predict(X_test_scaled)
    y_prob = svm_model.predict_proba(X_test_scaled)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)
    
    return svm_model, auc_score, report

def main():
    print("1. Generating Synthetic LMS Data...")
    df = generate_synthetic_lms_data(num_students=1200)
    
    print("2. Engineering CAPIR Behavioral Features...")
    df = calculate_capir_features(df)
    capir_features = df[['C', 'A', 'P', 'I', 'R']]
    
    print("3. Performing Behavioral Clustering (K=4)...")
    df['Cluster'] = perform_clustering(capir_features)
    print("   Cluster distribution:\n", df['Cluster'].value_counts().to_string())
    
    print("4. Training Early Warning System (EWS) Predictive Model...")
    # Using CAPIR scores to predict 'is_at_risk' status
    X = capir_features
    y = df['is_at_risk']
    
    model, auc, report = train_predictive_model(X, y)
    
    print(f"\n--- Model Evaluation ---")
    print(f"ROC-AUC Score: {auc:.3f}")
    print("Classification Report:\n", report)
    print("Pipeline Execution Complete. Ready for Early Warning System deployment.")

if __name__ == "__main__":
    main()