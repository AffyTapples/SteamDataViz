# visualization.py
# Only Plotly scatter plot – pure function, no Streamlit logic.
import pandas as pd
import plotly.express as px


def plot_clusters(df, embeddings, labels, title, search_term=''):
    plot_df = pd.DataFrame({
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'cluster': labels.astype(str),
        'name': df['name'],
        'genres': df['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
    })

    plot_df['highlight'] = plot_df['name'].str.contains(search_term, case=False) if search_term else False
    plot_df['marker_size'] = plot_df['highlight'].map({True: 20, False: 7})
    plot_df['color'] = plot_df.apply(lambda row: 'selected_game' if row['highlight'] else row['cluster'], axis=1)

    color_labels = {'selected_game': search_term} if search_term else {}

    fig = px.scatter(
        plot_df,
        x='x', y='y',
        color='color',
        size='marker_size',
        hover_data=['name', 'genres'],
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels=color_labels,
        title=title
    )
    return fig