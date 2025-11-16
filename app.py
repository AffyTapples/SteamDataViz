# app.py
# Streamlit UI – imports everything, handles layout, widgets, session state.

import os
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from analysis import (SHAP_AVAILABLE, _calculate_surrogate_importance_linear,
                      calculate_cluster_purity_with_majority_genre,
                      calculate_silhouette_score,
                      compare_silhouette_with_genres,
                      compare_silhouette_with_top_genres, compute_clustering,
                      compute_dr, get_feature_importance_pca,
                      train_genre_classifier)
from preprocessing import preprocess_data
from sklearn.metrics.pairwise import euclidean_distances
from visualization import plot_clusters


# ----------------- Image helper -----------------
@st.cache_data
def show_selected_image(_df, selected_game):
    if not selected_game:
        return
    try:
        os.makedirs('img', exist_ok=True)
        app_id = _df[_df['name'] == selected_game]['appID'].iloc[0]
        image_url = f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{app_id}/header.jpg"
        local_path = f'img/{app_id}.jpg'
        if not os.path.isfile(local_path):
            img_data = requests.get(image_url).content
            with open(local_path, 'wb') as f:
                f.write(img_data)
        st.image(local_path)
    except Exception as e:
        st.warning(f"Image error: {e}")


# ----------------- Page config & style -----------------
st.set_page_config(layout='wide', page_title="Steam Games Clustering Explorer", initial_sidebar_state="expanded")
st.markdown("""<style> ... (exact same CSS as original) ... </style>""", unsafe_allow_html=True)
st.markdown('<h1 class="main-header">Steam Games Clustering Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover patterns in your game collection using advanced ML techniques</p>', unsafe_allow_html=True)


# ----------------- Load data (cached) -----------------
with st.spinner("Loading & preprocessing data (cached)..."):
    df, X, feature_names, tfidf, df_genres_onehot, df_tags_onehot = preprocess_data(
        max_sample=30000, tfidf_max_features=300)


# ----------------- Tabs -----------------
tab1, tab2 = st.tabs(["Clustering Analysis", "Genre Classification"])

with tab1:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Configuration")
    dr_method = st.sidebar.selectbox(
    'Dimensionality Reduction',
    ['umap','pca','svd','tsne','mds','isomap'],
    index=0
            )
    clustering_method = st.sidebar.selectbox('Clustering Technique',['kmeans','agglomerative','dbscan'],index=0)
    st.sidebar.markdown("---")

    dr_blurbs = {
        'umap': (
            "UMAP: Builds a graph of nearest neighbors in high-dimensional space, then optimizes a low-dimensional layout "
            "to preserve both local and some global structure using stochastic gradient descent."
        ),
        'pca': (
            "PCA: Computes orthogonal axes (principal components) that capture maximal variance in the data, "
            "projecting points onto the first two components for dimensionality reduction."
        ),
        'tsne': (
            "t-SNE: Converts pairwise distances to probabilities reflecting similarity in high-dimensional space, "
            "then iteratively positions points in 2D to match those probabilities, emphasizing local neighborhoods."
        ),
        'mds': (
            "MDS: Uses pairwise distances to place points in low-dimensional space such that the distances between points "
            "match the original high-dimensional distances as closely as possible."
        ),
        'isomap': (
            "Isomap: Computes geodesic distances along a graph of nearest neighbors and uses MDS to embed points in 2D, "
            "preserving manifold structure in the data."
        ),
        'svd': (
            "SVD: SVD is a linear dimensionality reduction technique that factorizes a data matrix into singular vectors and singular values. It identifies the directions of maximum variance in the data and can be used for noise reduction, data compression, and feature extraction."
        )
    }

    cluster_blurbs = {
        'kmeans': (
            "KMeans: Randomly initializes k centroids, assigns points to the nearest centroid, "
            "then iteratively updates centroids to minimize intra-cluster squared distance until convergence."
        ),
        'agglomerative': (
            "Agglomerative: Starts with each point as its own cluster, then repeatedly merges the two closest clusters "
            "based on a linkage criterion, forming a hierarchical tree structure until the desired number of clusters is reached."
        ),
        'dbscan': (
            "DBSCAN: Groups points into clusters based on density. Points with at least `min_samples` neighbors within `eps` "
            "distance form a cluster; points not meeting this criterion are labeled as outliers."
        )
    }


    params = {}
    st.sidebar.markdown("### Parameters")
    if clustering_method in ['kmeans','agglomerative']:
        n_clusters = st.sidebar.slider('Number of clusters (k)',2,20,5)
        params['n_clusters'] = n_clusters
    else:
        eps = st.sidebar.slider('DBSCAN eps',0.1,2.0,0.5)
        min_samples = st.sidebar.slider('DBSCAN min_samples',2,20,5)
        params['eps'] = eps
        params['min_samples'] = min_samples

    st.sidebar.markdown("---")
    with st.sidebar.expander("Algorithm Details"):
        st.markdown(f"**{dr_method.upper()}**")
        st.caption(dr_blurbs[dr_method])
        st.markdown(f"**{clustering_method.upper()}**")
        st.caption(cluster_blurbs[clustering_method])

    with st.spinner("Computing DR and clustering..."):
        X_2d, dr_model = compute_dr(X, dr_method)
        labels = compute_clustering(X_2d, method=clustering_method, params=params)
        df['cluster'] = labels

    top_genre_comparison = None
    if clustering_method in ['kmeans', 'agglomerative'] and 'n_clusters' in params:
        top_genre_comparison = compare_silhouette_with_top_genres(X, labels, df, params['n_clusters'])

    sil_score = calculate_silhouette_score(X_2d, labels)
    # Compute cluster purity
    purity_score, cluster_info = calculate_cluster_purity_with_majority_genre(labels, df_genres_onehot, df_tags_onehot)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Games", f"{len(df):,}")
    with col2:
        unique_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        st.metric("Clusters Found", unique_clusters)
    with col3:
        outliers = np.sum(labels == -1)
        st.metric("Outliers", outliers)
    with col4:
        if sil_score is not None:
            st.metric("Silhouette Score", f"{sil_score:.3f}")
        else:
            avg_per_cluster = len(df) / unique_clusters if unique_clusters > 0 else 0
            st.metric("Avg per Cluster", f"{avg_per_cluster:.0f}")

    # Display purity
    if purity_score is not None:
        st.metric("Cluster Purity", f"{purity_score:.3f}")

    if cluster_info:
        cluster_purity_df = pd.DataFrame([
            {"Cluster": c, "Purity": info['purity'], "Majority Genre": info['majority_genre'], "Majority Tags": info['majority_tags']}
            for c, info in cluster_info.items()
        ]).sort_values("Cluster")
        st.dataframe(cluster_purity_df, use_container_width=True)
    st.markdown("---")

    if top_genre_comparison:
        st.markdown("### Cluster Quality vs. Top Genres + Other")
        col_comp1, col_comp2, col_comp3 = st.columns(3)
        with col_comp1:
            st.metric("Computed Silhouette", f"{top_genre_comparison['computed_silhouette']:.3f}" if top_genre_comparison['computed_silhouette'] is not None else "N/A")
        with col_comp2:
            st.metric("Top-Genre Silhouette", f"{top_genre_comparison['genre_silhouette']:.3f}")
        with col_comp3:
            st.metric("Better Separation", top_genre_comparison['better'])
        st.caption(f"Compares ML clusters to top {len(top_genre_comparison['top_genres_used'])} genres + 'Other'. Higher silhouette = better.")
        with st.expander("Top Genres Used"):
            st.write(top_genre_comparison['top_genres_used'])
    else:
        st.info("Top-genre comparison available only for KMeans/Agglomerative clustering.")

    selected_game_name = st.session_state.get('selected_game', None)
    highlight_name = selected_game_name if selected_game_name else ''
    title = f"{dr_method.upper()} + {clustering_method.upper()}"
    fig = plot_clusters(df, X_2d, labels, title, search_term=highlight_name)
    fig.update_layout(
        height=700,
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        title_font_size=20,
        title_x=0.5
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## Find Similar Games")
    search_col1, search_col2 = st.columns([2, 1])
    with search_col1:
        search_term = st.text_input("Search game title", "", placeholder="Type to filter games...", key="search_input")
    with search_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if search_term:
            match_count = df[df['name'].str.contains(search_term, case=False)]['name'].count()
            st.caption(f"{match_count} matches")

    if search_term:
        matches = df[df['name'].str.contains(search_term, case=False)]['name'].tolist()
    else:
        matches = df['name'].tolist()

    if matches:
        selected_game = st.selectbox('Select Game', matches, key='selected_game')
    else:
        selected_game = None
        st.info("No games found matching your search.")

    @st.cache_data
    def show_selected_image(selected_game):
        if not selected_game:
            return
            
        try:
            os.makedirs('img', exist_ok=True)
            appID = df[df['name'] == selected_game]['appID'].to_list()[0]
            image_url = f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appID}/header.jpg"
            local_path = f'img/{appID}.jpg'
            
            if not os.path.isfile(local_path):
                try:
                    img_data = requests.get(image_url).content
                    with open(local_path, 'wb') as handler:
                        handler.write(img_data)
                except Exception as e:
                    st.warning(f"Could not download image for {selected_game}: {e}")
                    st.image(image_url)
                    return

            if os.path.isfile(local_path):
                st.image(local_path)
                
        except Exception as e:
            st.warning(f"Error displaying image for {selected_game}: {e}")
            pass

    if selected_game:
        idx = df[df['name']==selected_game].index[0]
        game_cluster = df.loc[idx,'cluster']
    
    game_col1, game_col2 = st.columns([1, 2])
    with game_col1:
        st.markdown(f"### {selected_game}")
        st.markdown(f'<span class="cluster-badge">Cluster {game_cluster}</span>', unsafe_allow_html=True)
        game_data = df.iloc[idx]
        if pd.notna(game_data.get('metacritic_score')) and game_data.get('metacritic_score', 0) > 0:
            st.metric("Metacritic Score", int(game_data['metacritic_score']))
        if pd.notna(game_data.get('price')) and game_data.get('price', 0) > 0:
            st.metric("Price", f"${game_data['price']:.2f}")
        if isinstance(game_data.get('genres'), list) and game_data['genres']:
            st.markdown("**Genres:** " + ", ".join(game_data['genres'][:5]))
    
    with game_col2:
        show_selected_image(selected_game)
    
    st.markdown("---")
    st.markdown("### Similar Games")
    st.caption("Top 10 closest games in 2D embedding space")
    
    dists = euclidean_distances([X_2d[idx]], X_2d)[0]
    sorted_idx = np.argsort(dists)[1:11]
    similar = df.iloc[sorted_idx]['name'].tolist()
    websites = df.iloc[sorted_idx]['website'].tolist()
    distances = dists[sorted_idx]
    
    for i, (s, url, dist) in enumerate(zip(similar[:10], websites[:10], distances)):
        similar_col1, similar_col2 = st.columns([1, 4])
        with similar_col1:
            show_selected_image(s)
        with similar_col2:
            st.markdown(f"**{i+1}. [{s}]({url})**")
            st.caption(f"Distance: {dist:.3f}")
        if i < len(similar[:10]) - 1:
            st.markdown("---")
    

    if selected_game:
        st.markdown("---")
        with st.expander("Feature Importance Analysis"):
            with st.spinner("Computing feature importance..."):
                if dr_method in ['pca', 'svd']:
                    imp_df = get_feature_importance_pca(dr_model, feature_names)
                else:
                    imp_df = _calculate_surrogate_importance_linear(X, df['cluster'].values, tuple(feature_names), SHAP_AVAILABLE)
            st.dataframe(imp_df, use_container_width=True)
            st.caption("Top features driving cluster assignments")

    st.markdown("---")
    st.markdown("## Cluster Samples")
    st.caption("First 10 games from each cluster")

    cluster_groups = df.groupby('cluster')['name'].apply(list)
    valid_clusters = sorted([c for c in cluster_groups.index if c != -1])
    outliers = cluster_groups.get(-1, [])

    if valid_clusters:
        cols = st.columns(min(len(valid_clusters), 4))
        for idx, cluster in enumerate(valid_clusters[:8]):
            col_idx = idx % 4
            with cols[col_idx]:
                st.markdown(f'<div class="game-card"><h4>Cluster {cluster}</h4>', unsafe_allow_html=True)
                names = cluster_groups.loc[cluster][:10]
                for name in names:
                    game_url = df[df['name'] == name]['website'].values[0] if len(df[df['name'] == name]) > 0 else '#'
                    st.markdown(f"• [{name}]({game_url})")
                st.markdown('</div><br>', unsafe_allow_html=True)

    if outliers and len(outliers) > 0:
        st.markdown("### Outliers")
        outlier_names = outliers[:10]
        for name in outlier_names:
            game_url = df[df['name'] == name]['website'].values[0] if len(df[df['name'] == name]) > 0 else '#'
            st.markdown(f"- [{name}]({game_url})")

    st.markdown("---")
    with st.expander("Technical Notes"):
        st.markdown("""
        - **Memory Management**: TF-IDF max features and Top-N for one-hot can be reduced if you run into memory constraints
        - **SHAP Dependency**: SHAP is optional — if not installed, the app falls back to RF feature importances
        - **Sparse Matrices**: For large datasets, keep sparse matrices and use sparse-compatible DR (UMAP supports sparse)
        - **Performance**: Clustering and DR are cached for faster re-computation
        """)

with tab2:
    st.markdown("## Genre Classification")
    st.markdown("Train a Random Forest classifier to predict whether a game belongs to a specific genre.")
    
    all_genres = set()
    for genres_list in df['genres']:
        if isinstance(genres_list, list):
            all_genres.update(genres_list)
    
    genre_options = sorted(list(all_genres))
    
    if len(genre_options) == 0:
        st.warning("No genres available in the dataset")
    else:
        with st.spinner("Computing clusters for Silhouette Score..."):
            X_2d_genre = compute_dr(X, method='umap')[0]
            cluster_labels_genre = compute_clustering(X_2d_genre, method='kmeans', params={'n_clusters': 5})
            sil_score_genre = calculate_silhouette_score(X_2d_genre, cluster_labels_genre)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            target_genre = st.selectbox("Select genre to predict:", genre_options, index=min(0, len(genre_options)-1))
        
        with col2:
            genre_count = sum(1 for genres_list in df['genres'] if isinstance(genres_list, list) and target_genre in genres_list)
            st.metric("Games with this genre", genre_count)
        
        st.markdown("---")
        st.markdown("### Cluster Quality")
        if sil_score_genre is not None:
            st.metric("Silhouette Score", f"{sil_score_genre:.3f}")
            st.caption("Measures how well-separated clusters are. Values range from -1 to 1, with higher values indicating better-defined clusters.")
        else:
            st.info("Silhouette Score cannot be calculated with current clustering configuration.")
        
        if 'classification_run' not in st.session_state:
            st.session_state.classification_run = False
        if 'classification_results' not in st.session_state:
            st.session_state.classification_results = None
        if 'target_genre_last' not in st.session_state:
            st.session_state.target_genre_last = None
        
        if st.session_state.target_genre_last != target_genre:
            st.session_state.classification_run = False
            st.session_state.classification_results = None
            st.session_state.target_genre_last = target_genre
        
        if st.button("Analyze Selected Genre", type="primary"):
            with st.spinner(f"Training Random Forest classifier for '{target_genre}'..."):
                clf, accuracy, precision, recall, feature_importance, test_data = train_genre_classifier(
                    X, df, target_genre, tuple(feature_names)
                )
                
                if clf is None:
                    st.error(f"Not enough data to train classifier for '{target_genre}'. Need at least 10 samples.")
                    st.session_state.classification_run = False
                else:
                    st.session_state.classification_run = True
                    st.session_state.classification_results = {
                        'clf': clf,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'feature_importance': feature_importance,
                        'test_data': test_data,
                        'target_genre': target_genre
                    }
                    st.session_state.target_genre_last = target_genre
        
        if st.session_state.classification_run and st.session_state.classification_results:
            results = st.session_state.classification_results
            target_genre_display = results['target_genre']
            
            st.success("Model trained successfully!")
            
            st.markdown("---")
            st.markdown("### Model Performance")
            
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            with perf_col1:
                st.metric("Accuracy", f"{results['accuracy']:.3f}")
            with perf_col2:
                st.metric("Precision", f"{results['precision']:.3f}")
            with perf_col3:
                st.metric("Recall", f"{results['recall']:.3f}")
            
            st.caption(f"""
            - **Accuracy**: Overall correctness of the model
            - **Precision**: Of games predicted as '{target_genre_display}', what percentage were correct
            - **Recall**: Of all '{target_genre_display}' games, what percentage did we find
            """)
            
            st.markdown("---")
            st.markdown("### Feature Importance")
            st.markdown(f"Top features that help predict '{target_genre_display}' games")
            
            import plotly.express as px
            fig = px.bar(
                results['feature_importance'].head(15),
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Top 15 Features for Predicting '{target_genre_display}'",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                height=600,
                yaxis={'categoryorder':'total ascending'},
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("View all top 20 features"):
                st.dataframe(results['feature_importance'], use_container_width=True)
            
            st.markdown("---")
            find_subgenres = st.checkbox("Find Sub-Genres (Clustering)", value=False)
            
            if find_subgenres:
                genre_mask = df['genres'].apply(
                    lambda x: 1 if isinstance(x, list) and target_genre_display in x else 0
                )
                genre_indices = np.where(genre_mask == 1)[0]
                
                if len(genre_indices) < 10:
                    st.warning(f"Not enough games in '{target_genre_display}' for sub-genre clustering. Need at least 10 games.")
                else:
                    df_subset = df.iloc[genre_indices].reset_index(drop=True)
                    X_subset = X[genre_indices]
                    
                    st.markdown("---")
                    st.markdown("### Sub-Genre Clustering")
                    st.markdown(f"Clustering games within '{target_genre_display}' to discover sub-genres")
                    
                    n_subclusters = st.slider("Number of sub-clusters", min_value=2, max_value=10, value=5,
                                             key="subcluster_slider",
                                             help="Choose the number of sub-genre clusters to find within the selected genre")
                    
                    with st.spinner(f"Computing sub-genre clusters for '{target_genre_display}'..."):
                        X_subset_2d, dr_model_subset = compute_dr(X_subset, method='umap')
                        
                        subcluster_labels = compute_clustering(
                            X_subset_2d,
                            method='kmeans',
                            params={'n_clusters': n_subclusters}
                        )
                        
                        sil_score_sub = calculate_silhouette_score(X_subset_2d, subcluster_labels)
                        
                        st.markdown("---")
                        st.markdown(f"### Sub-Genre Cluster Results")
                        
                        sub_col1, sub_col2, sub_col3 = st.columns(3)
                        with sub_col1:
                            st.metric("Games Analyzed", f"{len(df_subset):,}")
                        with sub_col2:
                            st.metric("Sub-Clusters Found", n_subclusters)
                        with sub_col3:
                            if sil_score_sub is not None:
                                st.metric("Silhouette Score", f"{sil_score_sub:.3f}")
                            else:
                                st.metric("Silhouette Score", "N/A")
                        
                        title_sub = f"Sub-Genre Clusters found within '{target_genre_display}'"
                        fig_sub = plot_clusters(df_subset, X_subset_2d, subcluster_labels, title_sub, search_term='')
                        fig_sub.update_layout(
                            height=700,
                            template='plotly_white',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(size=12),
                            title_font_size=20,
                            title_x=0.5
                        )
                        st.plotly_chart(fig_sub, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("### Sub-Genre Cluster Profiles")
                        st.caption("Top features characterizing each sub-genre cluster")
                        
                        with st.spinner("Analyzing sub-genre cluster features..."):
                            df_subset_copy = df_subset.copy()
                            df_subset_copy['subcluster'] = subcluster_labels
                            
                            for cluster_id in sorted(set(subcluster_labels)):
                                if cluster_id == -1:
                                    continue
                                cluster_mask = subcluster_labels == cluster_id
                                cluster_df = df_subset_copy[cluster_mask]
                                
                                all_tags = []
                                for tags in cluster_df['tags']:
                                    if isinstance(tags, list):
                                        all_tags.extend([str(t) for t in tags])
                                    elif isinstance(tags, dict):
                                        all_tags.extend([str(t) for t in tags.keys()])
                                
                                if all_tags:
                                    tag_counts = Counter(all_tags)
                                    top_tags = tag_counts.most_common(10)
                                    
                                    st.markdown(f"**Sub-Cluster {cluster_id}** ({np.sum(cluster_mask)} games)")
                                    tag_str = ", ".join([f"{tag} ({count})" for tag, count in top_tags])
                                    st.caption(tag_str)
                                    
                                    sample_games = cluster_df['name'].head(5).tolist()
                                    if sample_games:
                                        st.caption(f"Sample games: {', '.join(sample_games)}")
                                
                                valid_clusters = sorted([c for c in set(subcluster_labels) if c != -1])
                                if cluster_id != valid_clusters[-1]:
                                    st.markdown("---")
                            
                            try:
                                imp_df_sub = _calculate_surrogate_importance_linear(
                                    X_subset,
                                    subcluster_labels,
                                    tuple(feature_names),
                                    SHAP_AVAILABLE
                                )
                                
                                if 'Note' not in imp_df_sub.columns:
                                    st.markdown("---")
                                    st.markdown("### Overall Feature Importance for Sub-Clusters")
                                    st.dataframe(imp_df_sub, use_container_width=True)
                                    st.caption("Top features that distinguish between the sub-genre clusters")
                            except Exception as e:
                                pass