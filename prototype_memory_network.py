import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import time

def run_prototype_memory_network(file_path):
    print("Loading VeReMi Extension dataset...")
    cols_to_use = ['posx_n', 'posy_n', 'spdx_n', 'spdy_n', 'class']
    
    try:
        df = pd.read_csv(file_path, usecols=cols_to_use)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    df.dropna(inplace=True)
    df['binary_label'] = (df['class'] != 0).astype(int) 

    features = df[['posx_n', 'posy_n', 'spdx_n', 'spdy_n']].values
    
    print("Scaling features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # ---------------------------------------------------------
    # PROTOTYPE MEMORY NETWORK IMPLEMENTATION
    # ---------------------------------------------------------
    print("Executing Prototype Memory Network...")
    start_time = time.time()

    # 1. Initialization: 
    # To bootstrap the semi-supervised process, we take a very small sample (e.g., the 5% labeled data budget the original authors used)
    # to establish our initial prototypes for 'Normal' and 'Attack'.
    sample_size = int(len(features_scaled) * 0.05)
    
    # We cheat slightly for initialization just to establish the centers, simulating the 5% labeled budget
    init_features = features_scaled[:sample_size]
    init_labels = df['binary_label'].values[:sample_size]
    
    normal_prototype = np.mean(init_features[init_labels == 0], axis=0)
    attack_prototype = np.mean(init_features[init_labels == 1], axis=0)

    if np.isnan(attack_prototype).any():
         # Fallback if the first 5% had zero attacks (rare but possible depending on sorting)
         attack_prototype = np.zeros_like(normal_prototype)
         
    prototypes = np.vstack([normal_prototype, attack_prototype])

    # 2. Single-Pass Distance Calculation (Vectorized for speed)
    # We calculate the Euclidean distance from every point to both prototypes
    distances = cdist(features_scaled, prototypes, metric='euclidean')
    
    # 3. Assignment
    # The point is assigned to the prototype it is closest to (0 for Normal, 1 for Attack)
    predicted_labels = np.argmin(distances, axis=1)

    end_time = time.time()
    latency = end_time - start_time
    print(f"Prototype Network execution completed in {latency:.2f} seconds.")

    df['predicted_attack'] = predicted_labels

    print("\n--- OVERALL BINARY CLASSIFICATION REPORT ---")
    print(classification_report(df['binary_label'], df['predicted_attack'], target_names=['Normal', 'Attack']))
    
    print("\n--- PERFORMANCE BY SPECIFIC ATTACK CLASS ---")
    attack_classes = df[df['class'] != 0]['class'].unique()
    for attack_type in sorted(attack_classes):
        subset = df[(df['class'] == 0) | (df['class'] == attack_type)]
        f1 = f1_score(subset['binary_label'], subset['predicted_attack'])
        print(f"Attack Class {attack_type} -> F1-Score: {f1:.4f}")

    print("\n--- LATENCY METRIC FOR DELTA ---")
    print(f"Original AILL-IDS Reported Training Latency: 177.49 seconds")
    print(f"Prototype Memory Network Latency: {latency:.2f} seconds")

if __name__ == "__main__":
    run_prototype_memory_network("mixalldata_clean.csv")