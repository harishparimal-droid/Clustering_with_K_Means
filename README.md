# Clustering_with_K_Means
This Python code performs K-Means clustering on the Mall_Customers dataset, using the elbow method to find the optimal number of clusters. It visualizes clusters based on income and spending score, evaluates them with silhouette score, and optionally uses PCA for multi-feature visualization and cluster analysis.
Mall Customers K-Means Clustering
Project Overview
This project performs unsupervised learning using K-Means clustering on the Mall Customers dataset. The goal is to segment customers based on annual income and spending score.

Features
Data loading and preprocessing

Elbow method for optimal cluster count

K-Means clustering and cluster assignment

Silhouette score evaluation

Visualizations of clusters with and without PCA

Usage
Run the provided Python script kmeans_clustering.py (or notebook) to reproduce the analysis and visualizations.

Dataset
The dataset Mall_Customers.csv contains customer demographics including Gender, Age, Annual Income, and Spending Score.

Results
Optimal clusters found using the elbow method

Cluster insights based on customer income and spending behavior

Evaluation using silhouette score to measure cluster quality

Future Work
Experiment with additional features for clustering

Test other clustering algorithms

Build a recommendation system based on clusters
