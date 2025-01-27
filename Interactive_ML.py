import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Put python -m streamlit run c:\Users\Shua\Documents\GitHub\SA2-Shuaaa-j\Interactive_ML.py in terminal to run

# Function to load and preprocess data
def load_data(file):
    data = pd.read_csv(file)
    data.columns = data.columns.str.strip()  # Clean column names
    return data

# Function for clustering and evaluation
def perform_clustering(X_train, y_train, method='KMeans', n_clusters=2):
    if method == 'KMeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'Hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')

    labels = model.fit_predict(X_train)

    # Evaluation Metrics
    silhouette = silhouette_score(X_train, labels)
    ch_score = calinski_harabasz_score(X_train, labels)
    db_score = davies_bouldin_score(X_train, labels)

    return labels, silhouette, ch_score, db_score

# Function for plotting confusion matrix
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    conf_matrix = confusion_matrix(y_true, y_pred)  # Correctly calculate confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', linewidths=0.5)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    st.pyplot(fig)

# Streamlit interface
st.title("Network Intrusion Detection using K-Means and Hierarchical Clustering")

# File upload for user to load dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    data = load_data(uploaded_file)

    # Display the dataset
    st.write("### Dataset Preview", data.head())

    # Separate features and label
    X = data.drop(columns=["Label"])
    y = data["Label"]

    # Handle missing values and scale data
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)

    # Split dataset for evaluation purposes
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Select clustering method
    clustering_method = st.selectbox('Choose Clustering Method', ['KMeans', 'Hierarchical'])

    # Choose number of clusters
    n_clusters = st.slider('Select Number of Clusters', min_value=2, max_value=5, value=2)

    # Perform Clustering
    labels, silhouette, ch_score, db_score = perform_clustering(X_train, y_train, method=clustering_method, n_clusters=n_clusters)

    # Display evaluation metrics
    st.write(f"### Clustering Evaluation Metrics ({clustering_method})")
    st.write(f"Silhouette Score: {silhouette:.4f}")
    st.write(f"Calinski-Harabasz Index: {ch_score:.4f}")
    st.write(f"Davies-Bouldin Index: {db_score:.4f}")

    # Plot Confusion Matrix
    st.write("### Confusion Matrix")
    plot_confusion_matrix(y_train, labels)

    # Visualization of Clusters (PCA 2D)
    st.write("### Clusters Visualization (PCA 2D)")
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_train)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=labels, cmap='viridis', edgecolors='k', alpha=0.7)
    plt.title(f"{clustering_method} Clusters (PCA 2D)")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()
    st.pyplot(plt)

    # Show PCA Explained Variance Ratio
    st.write("### Explained Variance Ratio of PCA Components")
    st.write(pca.explained_variance_ratio_)

