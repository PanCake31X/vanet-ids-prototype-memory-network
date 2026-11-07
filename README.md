# Semi-Supervised Prototype Memory Network for VANET IDS

This repository contains the evaluation scripts and graph generation code for our research paper: **"Overcoming Computational Latency in Semi-Supervised VANET Intrusion Detection Systems."**

## Overview
This project evaluates the computational bottlenecks of density-based stream clustering (DBSTREAM) used in state-of-the-art VANET Intrusion Detection Systems like AILL-IDS. We propose and evaluate a single-pass **Prototype Memory Network** to eliminate the dual-ensemble retraining latency while maintaining classification parity on raw spatial features.

## Files in this Repository
* `baseline_kmeans.py`: Replicates the baseline spatial clustering separation used prior to DBSTREAM density mapping.
* `prototype_memory_network.py`: Implements our proposed single-pass $O(m)$ Euclidean nearest-prototype assignment.
* `plot_results.py`: Matplotlib script to generate the computational latency comparison graphs used in the paper.

## Dataset
Evaluated on the **VeReMi Extension Dataset** (3,194,808 instances) targeting low-margin spatial falsifications (Constant and Random Offset attacks) under LuST traffic conditions.

## Key Results
* **Computational Latency:** Reduced from 177.49s (Dual-Ensemble DBSTREAM) to **0.09s** (Prototype Memory Network).
* **F1-Score Parity:** Maintained identical binary classification parity (F1=0.43) on raw spatial features, proving density-based micro-clustering adds severe computational overhead without measurable binary detection benefit in the spatial domain.