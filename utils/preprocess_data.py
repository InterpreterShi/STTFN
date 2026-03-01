#!/usr/bin/env python3
"""
Data preprocessing script for PeMS03 and PeMS07 datasets.
This script converts the raw distance.csv files to the format expected by STTFN.
"""
import os
import numpy as np
import pandas as pd

def process_distance_csv(input_file, output_file, num_nodes):
    """
    Convert distance.csv from (sensor_id, sensor_id, distance) format
    to (node_index, node_index, cost) format.
    """
    df = pd.read_csv(input_file)
    
    # Check column names and standardize
    if 'distance' in df.columns:
        df = df.rename(columns={'distance': 'cost'})
    
    # Get unique sensor IDs and create mapping to indices
    all_sensors = sorted(set(df['from'].unique()) | set(df['to'].unique()))
    sensor_to_idx = {sensor: idx for idx, sensor in enumerate(all_sensors)}
    
    print(f"Found {len(all_sensors)} unique sensors, expected {num_nodes}")
    
    # Convert sensor IDs to indices
    df['from'] = df['from'].map(sensor_to_idx)
    df['to'] = df['to'].map(sensor_to_idx)
    
    # Save processed file
    df.to_csv(output_file, index=False)
    print(f"Saved processed distance file to {output_file}")
    return df

def generate_adjacency_matrix_from_csv(distance_file, num_nodes, sigma=10, epsilon=0.5):
    """
    Generate adjacency matrix W_pemsXX.csv from distance.csv using Gaussian kernel.
    """
    df = pd.read_csv(distance_file)
    
    # Ensure correct column names
    if 'distance' in df.columns:
        df = df.rename(columns={'distance': 'cost'})
    
    # Get unique sensors and create mapping
    all_sensors = sorted(set(df['from'].unique()) | set(df['to'].unique()))
    sensor_to_idx = {sensor: idx for idx, sensor in enumerate(all_sensors)}
    
    # Create distance matrix
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for _, row in df.iterrows():
        i = sensor_to_idx.get(row['from'], row['from'])
        j = sensor_to_idx.get(row['to'], row['to'])
        if isinstance(i, int) and isinstance(j, int) and i < num_nodes and j < num_nodes:
            dist_matrix[i, j] = row['cost']
            dist_matrix[j, i] = row['cost']  # Symmetric
    
    # Apply Gaussian kernel: W_ij = exp(-d^2 / sigma^2) if d > 0, else 0
    W = np.zeros((num_nodes, num_nodes))
    distances = dist_matrix[dist_matrix > 0]
    if len(distances) > 0:
        std = distances.std()
        W = np.exp(-dist_matrix ** 2 / (std ** 2 + 1e-10))
        W[dist_matrix == 0] = 0
        np.fill_diagonal(W, 0)  # No self-loops
    
    return W

if __name__ == '__main__':
    import sys
    
    datasets = {
        'pems03': {'num_nodes': 358, 'path': 'dataset/pems03'},
        'pems07': {'num_nodes': 883, 'path': 'dataset/pems07'},
    }
    
    for name, info in datasets.items():
        distance_file = os.path.join(info['path'], 'distance.csv')
        if os.path.exists(distance_file):
            print(f"\nProcessing {name}...")
            
            # Process distance.csv (rename columns if needed)
            df = process_distance_csv(distance_file, distance_file, info['num_nodes'])
            
            # Generate adjacency matrix
            W = generate_adjacency_matrix_from_csv(distance_file, info['num_nodes'])
            W_file = os.path.join(info['path'], f'W_{name}.csv')
            pd.DataFrame(W).to_csv(W_file, index=False, header=False)
            print(f"Saved adjacency matrix to {W_file}, shape: {W.shape}")
        else:
            print(f"Warning: {distance_file} not found")
