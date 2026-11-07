import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import time

def run_baseline_evaluation(file_path):
    print("Loading VeReMi Extension dataset... (This may take a minute for 1.12 GB)")
    
    # Updated to match the exact schema of mixalldata_clean.csv
    # We use the _n columns because they contain the realistic sensor noise and attack perturbations
    cols_to_use = ['posx_n', 'posy_n', 'spdx_n', 'spdy_n', 'class']
    
    try:
        df = pd.read_csv(file_path, usecols=cols_to_use)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Dataset loaded. Total records: {len(df)}")
    
    # Clean data (drop NaNs if any)
    df.dropna(inplace=True)

    # 0 is Normal. Any other class is an Attack.
    df['binary_label'] = (df['class'] != 0).astype(int) 

    # Extract raw spatial and velocity features with noise/attacks applied
    features = df[['posx_n', 'posy_n', 'spdx_n', 'spdy_n']].values
    
    print("Scaling features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    print("Executing K-Means Clustering (k=2) to mimic baseline spatial separation...")
    start_time = time.time()
    
    # We use n_init='auto' to avoid warnings in newer sklearn versions
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    
    end_time = time.time()
    print(f"Clustering completed in {end_time - start_time:.2f} seconds.")

    # Align the arbitrary cluster ID (0 or 1) to the true label
    cluster_0_attack_rate = df[clusters == 0]['binary_label'].mean()
    if cluster_0_attack_rate > 0.5:
        predicted_labels = clusters
    else:
        predicted_labels = 1 - clusters

    df['predicted_attack'] = predicted_labels

    print("\n--- OVERALL BINARY CLASSIFICATION REPORT ---")
    print(classification_report(df['binary_label'], df['predicted_attack'], target_names=['Normal', 'Attack']))
    
    print("\n--- PERFORMANCE BY SPECIFIC ATTACK CLASS ---")
    attack_classes = df[df['class'] != 0]['class'].unique()
    for attack_type in sorted(attack_classes):
        subset = df[(df['class'] == 0) | (df['class'] == attack_type)]
        f1 = f1_score(subset['binary_label'], subset['predicted_attack'])
        print(f"Attack Class {attack_type} -> F1-Score: {f1:.4f}")

if __name__ == "__main__":
    run_baseline_evaluation("mixalldata_clean.csv")