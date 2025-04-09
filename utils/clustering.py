import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def preprocess_clustering_data(df, feature_columns):
    """Preprocess data for clustering."""
    X = df[feature_columns].copy()
    
    # Handle missing values
    for col in X.columns:
        X[col] = X[col].fillna(X[col].mean())
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, X_scaled, scaler

def perform_kmeans(X_scaled, n_clusters):
    """Perform K-means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    return kmeans, clusters

def plot_clusters(df, vis_features, clusters, kmeans, scaler):
    """Create a scatter plot of clusters."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each cluster
    for i in range(kmeans.n_clusters):
        cluster_data = df[df['cluster'] == i]
        ax.scatter(cluster_data[vis_features[0]], 
                   cluster_data[vis_features[1]], 
                   label=f'Cluster {i}')
    
    # Plot centroids
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='X', c='red', label='Centroids')
    
    ax.set_xlabel(vis_features[0])
    ax.set_ylabel(vis_features[1])
    ax.set_title('Cluster Visualization')
    ax.legend()
    
    return fig

def get_cluster_statistics(df, clusters, feature_columns, n_clusters):
    """Get statistics for each cluster."""
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    
    cluster_stats = {}
    for i in range(n_clusters):
        cluster_stats[i] = df_with_clusters[df_with_clusters['cluster'] == i][feature_columns].describe()
    
    return cluster_stats