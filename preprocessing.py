# preprocessing.py
# Loads games.json, cleans text, builds numeric + one-hot + TF-IDF features.
# Returns everything the app needs: df, X (dense matrix), feature_names, etc.

import json
import os
import re
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def load_games_json(path: str = 'games.json') -> pd.DataFrame:
    """Load raw JSON and convert to a clean DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    with open(path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    rows = []
    for app_id, game in dataset.items():
        tags_data = game.get('tags', {})
        tags = list(tags_data.keys()) if isinstance(tags_data, dict) else tags_data if isinstance(tags_data, list) else []

        rows.append({
            'appID': app_id,
            'name': game.get('name', '') or '',
            'release_date': game.get('release_date', None),
            'estimated_owners': game.get('estimated_owners', ''),
            'peak_ccu': game.get('peak_ccu', 0) or 0,
            'required_age': game.get('required_age', 0) or 0,
            'price': game.get('price', 0) or 0,
            'dlc_count': game.get('dlc_count', 0) or 0,
            'long_description': game.get('detailed_description', '') or '',
            'languages': game.get('supported_languages', []) or [],
            'windows': int(game.get('windows', False)),
            'mac': int(game.get('mac', False)),
            'linux': int(game.get('linux', False)),
            'metacritic_score': game.get('metacritic_score', 0) or 0,
            'user_score': game.get('user_score', 0) or 0,
            'positive': game.get('positive', 0) or 0,
            'negative': game.get('negative', 0) or 0,
            'achievements': game.get('achievements', 0) or 0,
            'recommendations': game.get('recommendations', 0) or 0,
            'average_playtime_forever': game.get('average_playtime_forever', 0) or 0,
            'developers': game.get('developers', []) or [],
            'publishers': game.get('publishers', []) or [],
            'categories': game.get('categories', []) or [],
            'genres': game.get('genres', []) or [],
            'tags': tags,
            'header_image': game.get('header_image', '') or '',
        })
    return pd.DataFrame(rows)


def clean_text_field(text):
    text = re.sub(r'<[^>]+>', ' ', str(text))
    text = text.replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def parse_owner_midpoint(s):
    nums = [int(n) for n in re.findall(r'\d+', str(s).replace(',', ''))]
    return np.mean(nums) if nums else np.nan


def one_hot_list_column_topN(df, col, top_n=50):
    df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
    freq = Counter([item for sub in df[col] for item in sub])
    top_values = set([item for item, _ in freq.most_common(top_n)])
    col_filtered = df[col].apply(lambda lst: [x for x in lst if x in top_values])
    col_str = col_filtered.apply(lambda x: ','.join(sorted(x)))
    if col_str.str.len().sum() == 0:
        return pd.DataFrame(index=df.index)
    one_hot = col_str.str.get_dummies(sep=',')
    one_hot.columns = [f"{col}_{c}" for c in one_hot.columns]
    return one_hot


def top_sorted_tags(tags):
    if isinstance(tags, dict):
        tags_list = list(tags.keys())
    elif isinstance(tags, list):
        tags_list = tags
    else:
        tags_list = []
    first3_sorted = sorted(tags_list[:3])
    return "_".join(first3_sorted)


def map_to_top20_fallback(tags_list, top20_tags):
    if not isinstance(tags_list, list) or len(tags_list) == 0:
        return 'Other'
    for combo in combinations(tags_list, 3):
        candidate = "_".join(combo)
        if candidate in top20_tags:
            return candidate
    top20_set = set(top20_tags)
    for combo in combinations(tags_list, 2):
        candidate_2 = "_".join(combo)
        for top_tag in top20_set:
            if candidate_2 in top_tag:
                return top_tag
    for tag in tags_list:
        for top_tag in top20_set:
            if tag in top_tag:
                return top_tag
    return 'Other'


def preprocess_data(max_sample=None, tfidf_max_features=300,
                    top_tags=10, top_genres=20, top_categories=20,
                    top_publishers=10, top_developers=10, top_languages=10):
    """Full preprocessing pipeline – returns everything the app needs."""
    df = load_games_json('games.json')

    # Normalise tags
    def normalize_tags(x):
        if isinstance(x, dict):
            return list(x.keys())
        elif isinstance(x, list):
            return [str(i) for i in x]
        elif pd.isna(x):
            return []
        else:
            return [str(x)]
    df['tags_clean'] = df['tags'].apply(normalize_tags)
    df['top_tags'] = df['tags_clean'].apply(top_sorted_tags)

    # Filter out adult content & non-ascii names & inactive games
    exclude_keywords = ['mature', 'nudity', 'sexual content']
    pattern = "|".join(exclude_keywords)
    df = df[~df['top_tags'].str.lower().str.contains(pattern)]
    df.reset_index(drop=True, inplace=True)
    df = df[df['name'].str.match(r'^[A-Za-z0-9\s\-\_\.,!:]+$', na=False)]
    df = df[df['peak_ccu'] > 0]

    # Dates & text
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['days_since_release'] = (pd.Timestamp.today() - df['release_date']).dt.days.fillna(0)
    df['clean_description'] = df['long_description'].astype(str).apply(clean_text_field)
    df['clean_description_short'] = df['clean_description'].apply(lambda s: s[:10000])
    df['estimated_owners_mid'] = df['estimated_owners'].apply(parse_owner_midpoint).fillna(0)
    df['website'] = 'https://store.steampowered.com/app/' + df['appID']

    # Log-transform skewed columns
    log_cols = ['peak_ccu', 'achievements', 'recommendations', 'average_playtime_forever',
                'estimated_owners_mid', 'positive', 'negative']
    df[log_cols] = np.log1p(df[log_cols].fillna(0))

    # Numeric features
    num_cols = [
        'days_since_release', 'peak_ccu', 'required_age', 'price', 'dlc_count',
        'metacritic_score', 'user_score', 'achievements', 'recommendations',
        'average_playtime_forever', 'estimated_owners_mid', 'positive', 'negative'
    ]
    df[num_cols] = df[num_cols].fillna(0)
    X_num = StandardScaler().fit_transform(df[num_cols])

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=tfidf_max_features, min_df=5, max_df=0.95, stop_words='english')
    X_text = tfidf.fit_transform(df['clean_description_short'].astype(str))

    # Weighted tags (original code)
    all_tags = df['tags_clean'].apply(pd.Series)
    if all_tags.shape[1] == 0:
        df_tags_flat = pd.DataFrame(index=df.index)
    else:
        df_tags_flat = pd.get_dummies(all_tags.stack()).groupby(level=0).sum()
        df_tags_flat = df_tags_flat.reindex(df.index, fill_value=0)
    if not df_tags_flat.empty:
        top_tag_cols = df_tags_flat.sum(axis=0).sort_values(ascending=False).head(top_tags).index
        df_tags_weighted = df_tags_flat[top_tag_cols].div(df_tags_flat.sum(axis=1).replace(0, 1), axis=0).fillna(0)
        df_tags_weighted.columns = [f"tags_{c}" for c in df_tags_weighted.columns]
    else:
        df_tags_weighted = pd.DataFrame(index=df.index)

    # Genres one-hot (top N)
    df_genres_onehot = one_hot_list_column_topN(df, 'genres', top_n=top_genres)

    # Top-tags → top20 with fallback
    top_tag_counts = df['top_tags'].value_counts()
    top20_tags = top_tag_counts.head(20).index
    df['top_tags_top20'] = df['top_tags'].apply(lambda x: x if x in top20_tags else 'Other')
    mask_other = df['top_tags_top20'] == 'Other'
    df.loc[mask_other, 'top_tags_top20'] = df.loc[mask_other, 'tags_clean'].apply(
        lambda tags: map_to_top20_fallback(tags, top20_tags) or 'Other'
    )
    df_top_tags_onehot = pd.get_dummies(df['top_tags_top20'], prefix='top_tags').astype(np.uint8)

    # Platforms
    X_platforms = df[['windows', 'mac', 'linux']].astype(int).values

    # Concatenate one-hots
    onehot_dfs = [df_genres_onehot, df_top_tags_onehot]
    onehot_dfs = [d if isinstance(d, pd.DataFrame) else pd.DataFrame(index=df.index) for d in onehot_dfs]
    X_onehot_df = pd.concat(onehot_dfs, axis=1).fillna(0)
    X_onehot = X_onehot_df.values if not X_onehot_df.empty else np.zeros((len(df), 0))

    # Final dense matrix (currently only one-hot; numeric & text are commented in original)
    X_dense = np.hstack([X_onehot])
    X_sparse = hstack([csr_matrix(X_dense)])

    # To dense (with memory fallback)
    try:
        X = X_sparse.toarray()
    except MemoryError:
        sample_n = min(30000, X_sparse.shape[0])
        X = X_sparse[:sample_n].toarray()
        df = df.iloc[:sample_n].reset_index(drop=True)

    df = df.reset_index(drop=True)
    df_genres_onehot = df_genres_onehot.reset_index(drop=True)
    df_top_tags_onehot = df_top_tags_onehot.reset_index(drop=True)

    feature_names = list(X_onehot_df.columns)

    if max_sample:
        df = df.head(max_sample)
        df_genres_onehot = df_genres_onehot.head(max_sample)
        df_top_tags_onehot = df_top_tags_onehot.head(max_sample)
        X = X[:max_sample]

    return df, X, feature_names, tfidf, df_genres_onehot, df_top_tags_onehot