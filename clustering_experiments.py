# analysis.py
# All ML heavy-lifting: DR, clustering, purity, feature importance, genre classifier.

from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import umap.umap_ as umap_
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def compute_dr(X, method='umap'):
    method = method.lower()
    if method == 'pca':
        dr = PCA(n_components=2, random_state=42)
    elif method == 'svd':
        dr = TruncatedSVD(n_components=2, random_state=42)
    elif method == 'tsne':
        dr = TSNE(n_components=2, random_state=42, perplexity=30, init='pca')
    elif method == 'umap':
        dr = umap_.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    elif method == 'mds':
        dr = MDS(n_components=2, random_state=42)
    elif method == 'isomap':
        dr = Isomap(n_components=2, n_neighbors=5)
    else:
        raise ValueError("Unsupported DR method")
    X_2d = dr.fit_transform(X)
    return X_2d, dr


def compute_clustering(embeddings, method='kmeans', params=None):
    params = params or {}
    method = method.lower()
    if method == 'kmeans':
        k = int(params.get('n_clusters', 5))
        return KMeans(n_clusters=k, random_state=42).fit_predict(embeddings)
    elif method == 'agglomerative':
        k = int(params.get('n_clusters', 5))
        return AgglomerativeClustering(n_clusters=k).fit_predict(embeddings)
    elif method == 'dbscan':
        eps = float(params.get('eps', 0.5))
        min_samples = int(params.get('min_samples', 5))
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(embeddings)
    else:
        raise ValueError("Unsupported clustering method")


def calculate_cluster_purity_with_majority_tags(labels, df_genres_onehot, df_top_tags_onehot, df_top_tags_subcategories_onehot):
    if df_genres_onehot.empty or df_top_tags_onehot.empty:
        return None, None
    cluster_info = {}
    total_correct = 0
    total_samples = 0
    for cluster in set(labels):
        if cluster == -1:
            continue
        idxs = np.where(labels == cluster)[0]
        cluster_genres = df_genres_onehot.iloc[idxs]
        cluster_tags = df_top_tags_onehot.iloc[idxs]
        cluster_tags_subs = df_top_tags_subcategories_onehot.iloc[idxs]
        if len(idxs) == 0:
            continue
        genre_sums = cluster_genres.sum(axis=0)
        tag_sums = cluster_tags.sum(axis=0)
        tag_subs_sums = cluster_tags_subs.sum(axis=0)
        majority_genre = genre_sums.idxmax()
        majority_tag = tag_sums.idxmax()
        majority_tag_subs = tag_subs_sums.idxmax()
        purity = genre_sums.max() / len(idxs)
        cluster_info[cluster] = {'purity': purity, 'majority_genre': majority_genre, 'majority_tags': majority_tag, 'majority_tags_subcategories': majority_tag_subs}
        total_correct += genre_sums.max()
        total_samples += len(idxs)
    purity_score = total_correct / total_samples if total_samples > 0 else None
    return purity_score, cluster_info

def compare_silhouette_with_top_genres(X, computed_labels, df, n_clusters):
    """
    Compute silhouette for computed clusters vs. top (n-1) genres + 'Other' pseudo-clusters.
    - X: Feature matrix.
    - computed_labels: Array of cluster labels.
    - df: DataFrame with 'genres' (list of strings).
    - n_clusters: The number of clusters (from params['n_clusters']).
    Returns: dict with scores or None if invalid.
    """
    if n_clusters < 2:
        return None  # Invalid for comparison
    
    # Step 1: Compute silhouette for computed clusters (filter noise)
    valid_idx = computed_labels != -1
    X_valid = X[valid_idx]
    if np.sum(valid_idx) < 2:
        computed_sil = None
    else:
        computed_sil = silhouette_score(X_valid, computed_labels[valid_idx])
    df_local = df.copy()
    # Step 2: Find top (n-1) genres by frequency
    all_genres = [
        genre
        for genres in df_local['genres'] if isinstance(genres, list)
        for genre in genres
    ]
    genre_counts = Counter(all_genres)
    top_genres = [g for g, _ in genre_counts.most_common(n_clusters - 1)]
    top_set = set(top_genres)

    # assign by highest-frequency genre (better than "first match")
    def assign_genre_cluster(genres):
        if not isinstance(genres, list):
            return 'Other'
        candidates = [g for g in genres if g in top_set]
        if not candidates:
            return 'Other'
        # pick most common genre among the game’s tags
        return max(candidates, key=lambda g: genre_counts[g])
    
    df_local['genre_cluster'] = df_local['genres'].apply(assign_genre_cluster)

    
    # Step 4: Encode to numeric labels (including 'Other')
    le = LabelEncoder()
    genre_labels = le.fit_transform(df_local['genre_cluster'])
    
    # Step 5: Compute silhouette for genre-based labels
    genre_sil = silhouette_score(X_valid, genre_labels[valid_idx])
    
    # Step 6: Compare and return
    # --- 4. Compare ---
    if computed_sil is None:
        better = "Genre-based"
    elif computed_sil > genre_sil:
        better = "Computed"
    elif genre_sil > computed_sil:
        better = "Genre-based"
    else:
        better = "Equal"
    result = {
        'computed_silhouette': computed_sil,
        'genre_silhouette': genre_sil,
        'better': better,
        'top_genres_used': top_genres  # Bonus: For display/debug
    }
    return result

def experiment_kmeans_purity_silhouette(
    X,
    df,
    df_genres_onehot,
    df_top_tags_onehot,
    df_top_tags_subcategories_onehot,
    k_min=5,
    k_max=30
):
    """
    Runs KMeans for cluster sizes k_min..k_max.
    Computes:
      - purity (from majority tags)
      - silhouette of KMeans clusters
      - silhouette of top-genre clusters
    Returns a pandas DataFrame with the results.
    """

    results = []

    for k in range(k_min, k_max + 1, 5):
        print(f"Running KMeans for k = {k}")

        # ---- 1. KMeans cluster assignment ----
        labels = compute_clustering(X, method='kmeans', params={'n_clusters': k})

        # ---- 2. Purity ----
        purity, cluster_info = calculate_cluster_purity_with_majority_tags(
            labels,
            df_genres_onehot,
            df_top_tags_onehot,
            df_top_tags_subcategories_onehot
        )

        # ---- 3. Cluster silhouette (may return None if invalid) ----
        try:
            cluster_sil = silhouette_score(X[labels != -1], labels[labels != -1]) \
                          if np.sum(labels != -1) > 1 else None
        except Exception:
            cluster_sil = None

        # ---- 4. Genre silhouette comparison ----
        genre_comp = compare_silhouette_with_top_genres(X, labels, df.copy(), k)
        if genre_comp is None:
            genre_sil = None
            better = None
        else:
            genre_sil = genre_comp['genre_silhouette']
            better = genre_comp['better']

        # ---- 5. Store results ----
        results.append({
            'k': k,
            'purity': purity,
            'cluster_silhouette': cluster_sil,
            'genre_silhouette': genre_sil,
            'genre_vs_cluster': better
        })

    return pd.DataFrame(results)


def experiment_umap_kmeans(
    X,
    df,
    df_genres_onehot,
    df_top_tags_onehot,
    df_top_tags_subcategories_onehot,
    n_clusters=24,
    dr_min=2,
    dr_max=10
):
    """
    Runs UMAP with dimensions dr_min..dr_max, then KMeans clustering with n_clusters.
    Computes purity, cluster silhouette, genre silhouette for each DR dimension.
    Returns a DataFrame with results.
    """
    results = []

    for dim in range(dr_min, dr_max + 1, 10):
        print(f"Running UMAP with n_components={dim}")

        # ---- 1. UMAP reduction ----
        umap_model = umap_.UMAP(n_components=dim, random_state=42, n_neighbors=15, min_dist=0.1)
        X_reduced = umap_model.fit_transform(X)

        # ---- 2. KMeans clustering ----
        labels = compute_clustering(X_reduced, method='kmeans', params={'n_clusters': n_clusters})

        # ---- 3. Purity ----
        purity, cluster_info = calculate_cluster_purity_with_majority_tags(
            labels,
            df_genres_onehot,
            df_top_tags_onehot,
            df_top_tags_subcategories_onehot
        )

        # ---- 4. Cluster silhouette ----
        try:
            cluster_sil = silhouette_score(X_reduced[labels != -1], labels[labels != -1]) \
                          if np.sum(labels != -1) > 1 else None
        except Exception:
            cluster_sil = None

        # ---- 5. Genre silhouette ----
        genre_comp = compare_silhouette_with_top_genres(X_reduced, labels, df.copy(), n_clusters)
        if genre_comp is None:
            genre_sil = None
            better = None
        else:
            genre_sil = genre_comp['genre_silhouette']
            better = genre_comp['better']

        # ---- 6. Store results ----
        results.append({
            'umap_dim': dim,
            'purity': purity,
            'cluster_silhouette': cluster_sil,
            'genre_silhouette': genre_sil,
            'genre_vs_cluster': better
        })

    return pd.DataFrame(results)