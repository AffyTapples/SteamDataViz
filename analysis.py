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


@st.cache_data
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


@st.cache_data
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


@st.cache_data
def calculate_cluster_purity_with_majority_genre(labels, df_genres_onehot, df_top_tags_onehot):
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
        if len(idxs) == 0:
            continue
        genre_sums = cluster_genres.sum(axis=0)
        tag_sums = cluster_tags.sum(axis=0)
        majority_genre = genre_sums.idxmax()
        majority_tag = tag_sums.idxmax()
        purity = genre_sums.max() / len(idxs)
        cluster_info[cluster] = {'purity': purity, 'majority_genre': majority_genre, 'majority_tags': majority_tag}
        total_correct += genre_sums.max()
        total_samples += len(idxs)
    purity_score = total_correct / total_samples if total_samples > 0 else None
    return purity_score, cluster_info


@st.cache_data
def _calculate_surrogate_importance(_X, _clusters, _feature_names_tuple, use_shap):
    # (tree-based version – kept for completeness, not used in UI)
    ...

# Linear surrogate (used in UI)
def _calculate_surrogate_importance_linear(_X, _clusters, _feature_names_tuple, use_shap):
    feature_names = list(_feature_names_tuple)
    valid_idx = _clusters != -1
    if np.sum(valid_idx) < 10 or len(np.unique(_clusters[valid_idx])) < 2:
        return pd.DataFrame({"Note": ["Insufficient clusters"]})

    n_valid = np.sum(valid_idx)
    subsample_size = min(max(200, int(n_valid * 0.2)), 30000)
    rng = np.random.default_rng(42)
    idxs = np.where(valid_idx)[0]
    subsample_idx = rng.choice(idxs, subsample_size, replace=False)

    X_subset = _X[subsample_idx]
    y_subset = _clusters[subsample_idx]

    enc = OneHotEncoder(sparse_output=False)
    y_oh = enc.fit_transform(y_subset.reshape(-1, 1))

    models = []
    for k in range(y_oh.shape[1]):
        lr = LinearRegression()
        lr.fit(X_subset, y_oh[:, k])
        models.append(lr)

    coef_mean = np.vstack([np.abs(m.coef_) for m in models]).mean(axis=0)
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": coef_mean}).sort_values("Importance", ascending=False)

    if use_shap:
        try:
            X_df = pd.DataFrame(X_subset, columns=feature_names)
            explainer = shap.LinearExplainer(models[0], X_df)
            shap_vals = explainer.shap_values(X_df)
            imp_df["Importance"] = np.abs(shap_vals).mean(axis=0)
        except Exception as e:
            return pd.DataFrame({"Note": [f"SHAP failed: {e}"]})

    return imp_df.sort_values("Importance", ascending=False).head(20)


def get_feature_importance_pca(_dr_model, feature_names):
    try:
        loadings = pd.DataFrame(np.abs(_dr_model.components_.T), index=feature_names, columns=['Dim1', 'Dim2'])
        top_dim1 = loadings['Dim1'].sort_values(ascending=False).head(10)
        top_dim2 = loadings['Dim2'].sort_values(ascending=False).head(10)
        return pd.concat([top_dim1, top_dim2], axis=1, keys=['Top Dim1', 'Top Dim2'])
    except Exception as e:
        return pd.DataFrame({"Note": [f"PCA loadings unavailable: {e}"]})


@st.cache_data
def calculate_silhouette_score(embeddings, labels):
    valid_idx = labels != -1
    if np.sum(valid_idx) < 10 or len(np.unique(labels[valid_idx])) < 2:
        return None
    try:
        return silhouette_score(embeddings[valid_idx], labels[valid_idx])
    except Exception:
        return None


@st.cache_data
def train_genre_classifier(_X, _df, target_genre, _feature_names_tuple):
    feature_names = list(_feature_names_tuple)
    df_copy = _df.copy()
    df_copy['has_genre'] = df_copy['genres'].apply(lambda x: 1 if isinstance(x, list) and target_genre in x else 0)
    if df_copy['has_genre'].sum() < 10:
        return None, None, None, None, None, None

    y = df_copy['has_genre'].values
    X_train, X_test, y_train, y_test = train_test_split(_X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False).head(20)

    return clf, accuracy, precision, recall, importance, (y_test, y_pred)


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
    if np.sum(valid_idx) < 2:
        computed_sil = None
    else:
        computed_sil = silhouette_score(X[valid_idx], computed_labels[valid_idx])
    
    # Step 2: Find top (n-1) genres by frequency
    all_genres = [genre for genres in df['genres'] if isinstance(genres, list) for genre in genres]
    genre_counts = Counter(all_genres)
    top_genres = [genre for genre, _ in genre_counts.most_common(n_clusters - 1)]
    top_set = set(top_genres)  # For fast lookup
    
    # Step 3: Assign each game to first matching top genre or 'Other'
    def assign_genre_cluster(genres):
        if not isinstance(genres, list):
            return 'Other'
        for genre in genres:
            if genre in top_set:
                return genre
        return 'Other'
    
    df['genre_cluster'] = df['genres'].apply(assign_genre_cluster)
    
    # Step 4: Encode to numeric labels (including 'Other')
    le = LabelEncoder()
    genre_labels = le.fit_transform(df['genre_cluster'])
    
    # Step 5: Compute silhouette for genre-based labels
    genre_sil = silhouette_score(X, genre_labels)
    
    # Step 6: Compare and return
    better = 'Computed' if (computed_sil is not None and computed_sil > genre_sil) else 'Genre-based' if genre_sil > (computed_sil or -float('inf')) else 'Equal'
    result = {
        'computed_silhouette': computed_sil,
        'genre_silhouette': genre_sil,
        'better': better,
        'top_genres_used': top_genres  # Bonus: For display/debug
    }
    return result