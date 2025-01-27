import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


# Load data (update with your dataset path)
data = pd.read_csv("CICIDS2017.csv")

# Clean column names
data.columns = data.columns.str.strip()

# Separate features and label
X = data.drop(columns=["Label"])
y = data["Label"]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Apply PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

print("PCA transformation successful.")

# Split dataset for evaluation purposes
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_train)

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=2, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_train)

# Evaluate Clustering (Unsupervised Metrics)
kmeans_silhouette = silhouette_score(X_train, kmeans_labels)
kmeans_ch_score = calinski_harabasz_score(X_train, kmeans_labels)
kmeans_db_score = davies_bouldin_score(X_train, kmeans_labels)

hierarchical_silhouette = silhouette_score(X_train, hierarchical_labels)
hierarchical_ch_score = calinski_harabasz_score(X_train, hierarchical_labels)
hierarchical_db_score = davies_bouldin_score(X_train, hierarchical_labels)

print("=== K-Means Clustering ===")
print(f"Silhouette Score: {kmeans_silhouette}")
print(f"Calinski-Harabasz Index: {kmeans_ch_score}")
print(f"Davies-Bouldin Index: {kmeans_db_score}")

print("=== Hierarchical Clustering ===")
print(f"Silhouette Score: {hierarchical_silhouette}")
print(f"Calinski-Harabasz Index: {hierarchical_ch_score}")
print(f"Davies-Bouldin Index: {hierarchical_db_score}")


# Visualize Silhouette Score (Side-by-Side Bar Plot)
metrics = ["Silhouette Score"]
kmeans_scores = [kmeans_silhouette]
hierarchical_scores = [hierarchical_silhouette]

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
x = np.arange(len(metrics))
width = 0.35

ax.bar(x - width/2, kmeans_scores, width, label='K-Means', color='blue')
ax.bar(x + width/2, hierarchical_scores, width, label='Hierarchical', color='orange')

ax.set_xlabel("Metrics")
ax.set_ylabel("Score")
ax.set_title("Silhouette Score Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.show()

# Visualize Calinski-Harabasz Index (Side-by-Side Bar Plot)
metrics = ["Calinski-Harabasz Index"]
kmeans_scores = [kmeans_ch_score]
hierarchical_scores = [hierarchical_ch_score]

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.bar(x - width/2, kmeans_scores, width, label='K-Means', color='blue')
ax.bar(x + width/2, hierarchical_scores, width, label='Hierarchical', color='orange')

ax.set_xlabel("Metrics")
ax.set_ylabel("Score")
ax.set_title("Calinski-Harabasz Index Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.show()

# Visualize Davies-Bouldin Index (Side-by-Side Bar Plot)
metrics = ["Davies-Bouldin Index"]
kmeans_scores = [kmeans_db_score]
hierarchical_scores = [hierarchical_db_score]

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.bar(x - width/2, kmeans_scores, width, label='K-Means', color='blue')
ax.bar(x + width/2, hierarchical_scores, width, label='Hierarchical', color='orange')

ax.set_xlabel("Metrics")
ax.set_ylabel("Score")
ax.set_title("Davies-Bouldin Index Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.show()

# Evaluate Clustering (Supervised Metrics using Ground Truth)
kmeans_accuracy = accuracy_score(y_train, kmeans_labels)
kmeans_precision = precision_score(y_train, kmeans_labels, average='weighted', zero_division=1)
kmeans_recall = recall_score(y_train, kmeans_labels, average='weighted', zero_division=1)
kmeans_f1 = f1_score(y_train, kmeans_labels, average='weighted')
kmeans_conf_matrix = confusion_matrix(y_train, kmeans_labels)

hierarchical_accuracy = accuracy_score(y_train, hierarchical_labels)
hierarchical_precision = precision_score(y_train, hierarchical_labels, average='weighted', zero_division=1)
hierarchical_recall = recall_score(y_train, hierarchical_labels, average='weighted', zero_division=1)
hierarchical_f1 = f1_score(y_train, hierarchical_labels, average='weighted')
hierarchical_conf_matrix = confusion_matrix(y_train, hierarchical_labels)

# Print text-only output for metrics
print("=== K-Means Clustering ===")
print(f"Accuracy: {kmeans_accuracy}")
print(f"Precision: {kmeans_precision}")
print(f"Recall: {kmeans_recall}")
print(f"F1-Score: {kmeans_f1}")
print(f"Confusion Matrix:\n{kmeans_conf_matrix}\n")

print("=== Hierarchical Clustering ===")
print(f"Accuracy: {hierarchical_accuracy}")
print(f"Precision: {hierarchical_precision}")
print(f"Recall: {hierarchical_recall}")
print(f"F1-Score: {hierarchical_f1}")
print(f"Confusion Matrix:\n{hierarchical_conf_matrix}\n")


# Visualize Supervised Metrics (Side-by-Side Bar Plot)
supervised_metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
kmeans_supervised_scores = [kmeans_accuracy, kmeans_precision, kmeans_recall, kmeans_f1]
hierarchical_supervised_scores = [hierarchical_accuracy, hierarchical_precision, hierarchical_recall, hierarchical_f1]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x = np.arange(len(supervised_metrics))

ax.bar(x - width/2, kmeans_supervised_scores, width, label='K-Means', color='blue')
ax.bar(x + width/2, hierarchical_supervised_scores, width, label='Hierarchical', color='orange')

ax.set_xlabel("Metrics")
ax.set_ylabel("Score")
ax.set_title("Supervised Metrics Comparison")
ax.set_xticks(x)
ax.set_xticklabels(supervised_metrics)
ax.legend()
plt.show()

# Visualize Confusion Matrices Side-by-Side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

kmeans_conf_matrix_df = pd.DataFrame(kmeans_conf_matrix, index=["True: 0", "True: 1"], columns=["Pred: 0", "Pred: 1"])
sns.heatmap(kmeans_conf_matrix_df, annot=True, fmt='g', cmap='Blues', ax=axes[0])
axes[0].set_title("K-Means Confusion Matrix")

hierarchical_conf_matrix_df = pd.DataFrame(hierarchical_conf_matrix, index=["True: 0", "True: 1"], columns=["Pred: 0", "Pred: 1"])
sns.heatmap(hierarchical_conf_matrix_df, annot=True, fmt='g', cmap='Oranges', ax=axes[1])
axes[1].set_title("Hierarchical Confusion Matrix")

plt.tight_layout()
plt.show()

# K-Means Clustering on PCA-transformed data
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_pca)

# Plot the K-Means clusters and centroids
plt.figure(figsize=(10, 7))

# Plot the data points color-coded by K-Means labels
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=10, alpha=0.5)

# Plot the centroids for K-Means
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label="Centroids")

plt.title("K-Means Clustering (PCA-transformed Data)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster Label')  # Add colorbar to show cluster label
plt.legend()
plt.show()

# Hierarchical Clustering on PCA-transformed data
hierarchical = AgglomerativeClustering(n_clusters=2, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_pca)

# Plot the Hierarchical Clustering blobs
plt.figure(figsize=(10, 7))

# Plot the data points color-coded by Hierarchical labels
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='rainbow', s=10, alpha=0.5)


plt.title("Hierarchical Clustering (PCA-transformed Data)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster Label')  # Add colorbar to show cluster label
plt.show()

# Visualize Dendrogram (for Hierarchical Clustering)
plt.figure(figsize=(10, 7))
plt.title("Dendrogram (Hierarchical Clustering)")
dendrogram(linkage(X_pca, method='ward'))
plt.show()

