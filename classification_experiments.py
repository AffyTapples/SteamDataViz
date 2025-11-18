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

def train_genre_classifier2(_X, _df, target_genre, _feature_names_tuple):
    feature_names = list(_feature_names_tuple)
    df_copy = _df.copy()
    df_copy['has_genre'] = df_copy['genres'].apply(lambda x: 1 if isinstance(x, list) and target_genre in x else 0)
    if df_copy['has_genre'].sum() < 10:
        return None, None, None, None, None, None, None
    genre_count = df_copy['has_genre'].sum()
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

    return clf, accuracy, precision, recall, importance, (y_test, y_pred),  genre_count

def run_genre_classification_suite(X, df, genre_list, feature_names_tuple):
    rows = []

    for genre in genre_list:
        clf, acc, prec, rec, importance, preds, genre_count = train_genre_classifier2(
            X, df, genre, feature_names_tuple
        )

        # Skip genres that return None (because <10 examples)
        if acc is None:
            continue

        rows.append({
            "genre": genre,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "genre_count": genre_count
        })

    return pd.DataFrame(rows)