# recommendation_models.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from annoy import AnnoyIndex
from scipy.sparse import csr_matrix
import implicit

# --- Recommendation #1 - Content Base Filtering based on 1 category (title) ---
def recommendations_by_title(dataframe, title_col, game_name, top_n=10):
    """
    Recommends games based on the similarity of their titles using TF-IDF and cosine similarity.
    """
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(dataframe[title_col])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(dataframe.index, index=dataframe[title_col]).drop_duplicates()
    try:
        idx = indices[game_name]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        game_indices = [i for i in sim_scores[1:top_n+1]]
        return dataframe['title'].iloc[game_indices]
    except KeyError:
        print(f"Game title '{game_name}' not found in the dataset.")
        return pd.Series()

# --- Recommendation #2 - Content Base Filtering based on 2 categories (genres and rating_score) ---
def content_based_recommendation_genres_rating(game_title, df, genre_weight=0.5, rating_score_weight=0.5, similarity_threshold=0.6, top_n=10):
    """
    Recommends games based on the similarity of their genres (using TF-IDF and cosine similarity)
    and normalized rating scores.
    Note: This function might require significant RAM for large datasets.
    """
    df['genres'] = df['genres'].astype(str)
    tfidf_genres = TfidfVectorizer(stop_words='english')
    tfidf_matrix_genres = tfidf_genres.fit_transform(df['genres'])
    cosine_sim_genres = cosine_similarity(tfidf_matrix_genres, tfidf_matrix_genres)
    df['rating_score_norm'] = (df['rating_score'] - df['rating_score'].min()) / (df['rating_score'].max() - df['rating_score'].min())
    similarity_matrix = (cosine_sim_genres * genre_weight) + (df['rating_score_norm'].values.reshape(-1, 1) * rating_score_weight)
    idx = df[df['title'] == game_title].index
    if len(idx) == 0:
        print(f"Game title '{game_title}' not found in the dataset.")
        return []
    else:
        idx = idx
        rated_games = set([game_title])
        recommendations = []
        for i, score in enumerate(similarity_matrix[idx]):
            if i != idx and df['title'].iloc[i] not in rated_games and score > similarity_threshold:
                recommendations.append((df['title'].iloc[i], score))
                rated_games.add(df['title'].iloc[i])
        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
        return recommendations

# --- Recommendation #3 - Content Base Filtering based on 3 categories with Annoy ---
def content_based_recommendation_genres_rating_metacritic_annoy(game_title, df, genre_weight=0.3, rating_score_weight=0.3, metacritic_score_weight=0.4, similarity_threshold=0.6, top_n=10, n_trees=10):
    """
    Recommends games based on genres, rating score, and Metacritic score using Annoy for efficient similarity search.
    """
    df['genres'] = df['genres'].astype(str)
    tfidf_genres = TfidfVectorizer(stop_words='english')
    tfidf_matrix_genres = tfidf_genres.fit_transform(df['genres'])
    df['rating_score'] = pd.to_numeric(df['rating_score'], errors='coerce')
    df['metacritic_score'] = pd.to_numeric(df['metacritic_score'], errors='coerce')
    df['rating_score_norm'] = (df['rating_score'] - df['rating_score'].min()) / (df['rating_score'].max() - df['rating_score'].min())
    df['metacritic_score_norm'] = (df['metacritic_score'] - df['metacritic_score'].min()) / (df['metacritic_score'].max() - df['metacritic_score'].min())
    combined_features = np.hstack([
        tfidf_matrix_genres.toarray() * genre_weight,
        df['rating_score_norm'].fillna(0).values.reshape(-1, 1) * rating_score_weight, # Handling NaN values
        df['metacritic_score_norm'].fillna(0).values.reshape(-1, 1) * metacritic_score_weight # Handling NaN values
    ])
    f = combined_features.shape[1]
    t = AnnoyIndex(f, 'angular')
    for i in range(len(combined_features)):
        t.add_item(i, combined_features[i])
    t.build(n_trees)
    idx = df[df['title'] == game_title].index
    if len(idx) == 0:
        print(f"Game title '{game_title}' not found in the dataset.")
        return []
    else:
        idx = idx
        similar_items = t.get_nns_by_item(idx, top_n + 1, include_distances=True)
        recommendations = [(df['title'].iloc[i], 1 - dist) for i, dist in zip(similar_items, similar_items[1]) if i != idx and 1 - dist > similarity_threshold]
        return recommendations

# --- Recommendation #4 - Content Base Filtering based on 4 categories with Annoy ---
def content_based_recommendation_all_features_annoy(game_title, df, genre_weight=0.25, rating_weight=0.25, rating_score_weight=0.25, metacritic_score_weight=0.25, similarity_threshold=0.6, top_n=10, n_trees=10):
    """
    Recommends games based on genres, general rating, rating score, and Metacritic score using Annoy.
    """
    df['genres'] = df['genres'].astype(str)
    tfidf_genres = TfidfVectorizer(stop_words='english')
    tfidf_matrix_genres = tfidf_genres.fit_transform(df['genres'])
    df['rating_score'] = pd.to_numeric(df['rating_score'], errors='coerce')
    df['metacritic_score'] = pd.to_numeric(df['metacritic_score'], errors='coerce')
    df['rating_score_norm'] = (df['rating_score'] - df['rating_score'].min()) / (df['rating_score'].max() - df['rating_score'].min())
    df['metacritic_score_norm'] = (df['metacritic_score'] - df['metacritic_score'].min()) / (df['metacritic_score'].max() - df['metacritic_score'].min())
    le = LabelEncoder()
    df['rating_encoded'] = le.fit_transform(df['rating'].astype(str)) # Ensure rating is string for encoding
    df['rating_encoded_norm'] = (df['rating_encoded'] - df['rating_encoded'].min()) / (df['rating_encoded'].max() - df['rating_encoded'].min())
    combined_features = np.hstack([
        tfidf_matrix_genres.toarray() * genre_weight,
        df['rating_encoded_norm'].fillna(0).values.reshape(-1, 1) * rating_weight, # Handling NaN values
        df['rating_score_norm'].fillna(0).values.reshape(-1, 1) * rating_score_weight, # Handling NaN values
        df['metacritic_score_norm'].fillna(0).values.reshape(-1, 1) * metacritic_score_weight # Handling NaN values
    ])
    f = combined_features.shape[1]
    t = AnnoyIndex(f, 'angular')
    for i in range(len(combined_features)):
        t.add_item(i, combined_features[i])
    t.build(n_trees)
    idx = df[df['title'] == game_title].index
    if len(idx) == 0:
        print(f"Game title '{game_title}' not found in the dataset.")
        return []
    else:
        idx = idx
        similar_items = t.get_nns_by_item(idx, top_n + 1, include_distances=True)
        recommendations = [(df['title'].iloc[i], 1 - dist) for i, dist in zip(similar_items, similar_items[1]) if i != idx and 1 - dist > similarity_threshold]
        return recommendations

# --- Recommendation #5 - Collaborative Filtering based on 1 category (total_playtime) ---
def collaborative_filtering_by_playtime(user_id, df, top_n=20, n_similar_users=5):
    """
    Recommends games based on user-user similarity calculated from total playtime.
    """
    user_item_matrix = pd.pivot_table(df, values='total_playtime', index='user_id', columns='title', fill_value=0)
    if user_id not in user_item_matrix.index:
        print(f"User ID '{user_id}' not found in the data.")
        return []
    user_item_matrix_filled = user_item_matrix.fillna(0)
    user_similarity = cosine_similarity(user_item_matrix_filled)
    user_index = user_item_matrix.index.get_loc(user_id)
    user_similarity_scores = user_similarity[user_index]
    top_similar_users_indices = np.argsort(user_similarity_scores)[::-1][1:n_similar_users+1]
    recommendations = set()
    for similar_user_index in top_similar_users_indices:
        similar_user_id = user_item_matrix.index[similar_user_index]
        played_games = df[df['user_id'] == similar_user_id]['title'].unique()
        recommendations.update(played_games)
    user_played_games = df[df['user_id'] == user_id]['title'].unique()
    recommendations = [game for game in recommendations if game not in user_played_games]
    return list(recommendations)[:top_n]

# --- Recommendation #6 - Collaborative Filtering based on 3 categories (rating, rating_score and total_playtime) ---
def collaborative_filtering_with_ratings(user_id, df, top_n=20, n_similar_users=5):
    """
    Recommends games based on user-user similarity calculated from a combined score of rating and total playtime.
    """
    rating_mapping = {
        'Overwhelmingly Positive': 5, 'Very Positive': 4, 'Positive': 3, 'Mostly Positive': 2, 'Mixed': 1,
        'Mostly Negative': 0, 'Negative': -1, 'Very Negative': -2, 'Overwhelmingly Negative': -3
    }
    df_copy = df.copy()
    df_copy['rating_score'] = df_copy['rating'].map(rating_mapping).fillna(df_copy['rating'].astype(str).astype('category').cat.codes) # Handle unseen ratings
    df_copy['total_playtime_norm'] = (df_copy['total_playtime'] - df_copy['total_playtime'].min()) / (df_copy['total_playtime'].max() - df_copy['total_playtime'].min())
    df_copy['rating_score_norm'] = (df_copy['rating_score'] - df_copy['rating_score'].min()) / (df_copy['rating_score'].max() - df_copy['rating_score'].min())
    df_copy['combined_score'] = df_copy['total_playtime_norm'] * 0.5 + df_copy['rating_score_norm'] * 0.5
    user_item_matrix = pd.pivot_table(df_copy, values='combined_score', index='user_id', columns='title', fill_value=0)
    if user_id not in user_item_matrix.index:
        print(f"User ID '{user_id}' not found in the data.")
        return []
    user_similarity = cosine_similarity(user_item_matrix.fillna(0))
    user_index = user_item_matrix.index.get_loc(user_id)
    user_similarity_scores = user_similarity[user_index]
    top_similar_users_indices = np.argsort(user_similarity_scores)[::-1][1:n_similar_users+1]
    recommendations = set()
    for similar_user_index in top_similar_users_indices:
        similar_user_id = user_item_matrix.index[similar_user_index]
        played_games = df_copy[df_copy['user_id'] == similar_user_id]['title'].unique()
        recommendations.update(played_games)
    user_played_games = df_copy[df_copy['user_id'] == user_id]['title'].unique()
    recommendations = [game for game in recommendations if game not in user_played_games]
    return list(recommendations)[:top_n]

# --- Recommendation #7 - Collaborative Filtering based on 3 categories (rating, rating_score, and total_playtime) with only the top 5 of the highest playing_time for each user ---
def collaborative_filtering_top_5_playtime(user_id, df, top_n=20, n_similar_users=5):
    """
    Recommends games based on user-user similarity using only the top 5 played games per user, considering rating.
    """
    def top_n_playtime(group, n=5):
        return group.nlargest(n, 'total_playtime')
    top_5_playtime_users = df.groupby('user_id', group_keys=False).apply(top_n_playtime).reset_index(drop=True)
    rating_mapping = {
        'Overwhelmingly Positive': 5, 'Very Positive': 4, 'Positive': 3, 'Mostly Positive': 2, 'Mixed': 1,
        'Mostly Negative': 0, 'Negative': -1, 'Very Negative': -2, 'Overwhelmingly Negative': -3
    }
    df_copy = top_5_playtime_users.copy()
    # Merge with df_games to get ratings if not already present in top_5_playtime_users
    if 'rating' not in df_copy.columns:
        # Assuming df_games is available in the scope where this function is called
        df_copy = pd.merge(df_copy, df_games[['app_id', 'rating']], on='app_id', how='left')
    df_copy['rating_score'] = df_copy['rating'].map(rating_mapping).fillna(df_copy['rating'].astype(str).astype('category').cat.codes) # Handle unseen ratings
    df_copy['total_playtime_norm'] = (df_copy['total_playtime'] - df_copy['total_playtime'].min()) / (df_copy['total_playtime'].max() - df_copy['total_playtime'].min())
    df_copy['rating_score_norm'] = (df_copy['rating_score'] - df_copy['rating_score'].min()) / (df_copy['rating_score'].max() - df_copy['rating_score'].min())
    df_copy['combined_score'] = df_copy['total_playtime_norm'] * 0.5 + df_copy['rating_score_norm'] * 0.5
    user_item_matrix = pd.pivot_table(df_copy, values='combined_score', index='user_id', columns='title', fill_value=0)
    if user_id not in user_item_matrix.index:
        print(f"User ID '{user_id}' not found in the data.")
        return []
    user_similarity = cosine_similarity(user_item_matrix.fillna(0))
    user_index = user_item_matrix.index.get_loc(user_id)
    user_similarity_scores = user_similarity[user_index]
    top_similar_users_indices = np.argsort(user_similarity_scores)[::-1][1:n_similar_users+1]
    recommendations = set()
    for similar_user_index in top_similar_users_indices:
        similar_user_id = user_item_matrix.index[similar_user_index]
        played_games = df_copy[df_copy['user_id'] == similar_user_id]['title'].unique()
        recommendations.update(played_games)
    user_played_games = df_copy[df_copy['user_id'] == user_id]['title'].unique()
    recommendations = [game for game in recommendations if game not in user_played_games]
    return list(recommendations)[:top_n]

# --- Recommendation #8 - Hybrid Model based on 4 categories (genres, rating, rating_score, and total_playtime) with only the top 10 of the highest playing_time for each user ---
def hybrid_recommendation_top_10_playtime(user_id, df_games, df_users, top_n=20, n_similar_users=5):
    """
    Hybrid recommendation using genres, rating, rating score, and top 10 played games per user.
    Note: This function might require significant RAM for large datasets.
    """
    def top_n_playtime(group, n=10):
        return group.nlargest(n, 'total_playtime')
    top_10_playtime_users = df_users.groupby('user_id', group_keys=False).apply(top_n_playtime).reset_index(drop=True)
    df_merged = pd.merge(df_games[['app_id', 'title', 'genres', 'rating']], top_10_playtime_users, on='app_id', how='inner')
    rating_mapping = {
        'Overwhelmingly Positive': 5, 'Very Positive': 4, 'Positive': 3, 'Mostly Positive': 2, 'Mixed': 1,
        'Mostly Negative': 0, 'Negative': -1, 'Very Negative': -2, 'Overwhelmingly Negative': -3
    }
    df_merged['rating_score'] = df_merged['rating'].map(rating_mapping).fillna(df_merged['rating'].astype(str).astype('category').cat.codes) # Handle unseen ratings
    df_merged['total_playtime_norm'] = (df_merged['total_playtime'] - df_merged['total_playtime'].min()) / (df_merged['total_playtime'].max() - df_merged['total_playtime'].min())
    df_merged['rating_score_norm'] = (df_merged['rating_score'] - df_merged['rating_score'].min()) / (df_merged['rating_score'].max() - df_merged['rating_score'].min())
    tfidf_genres = TfidfVectorizer(stop_words='english')
    df_merged['genres'] = df_merged['genres'].fillna('')
    tfidf_matrix_genres = tfidf_genres.fit_transform(df_merged.groupby('title')['genres'].first()) # Unique genres per title
    genre_features_df = pd.DataFrame(tfidf_matrix_genres.toarray(), index=df_merged['title'].unique())
    user_features = df_merged.pivot_table(index='user_id', columns='title', values=['total_playtime_norm', 'rating_score_norm'], fill_value=0)
    if user_id not in user_features.index:
        print(f"User ID '{user_id}' not found in the data.")
        return []
    user_row = user_features.loc[user_id].values.flatten()
    all_other_users_rows = user_features.drop(user_id, errors='ignore').values
    if all_other_users_rows.size == 0:
        return [] # No other users to compare with
    similarity_scores = cosine_similarity([user_row], all_other_users_rows)
    similar_user_indices = np.argsort(similarity_scores)[::-1][:n_similar_users]
    similar_users_ids = user_features.drop(user_id, errors='ignore').index[similar_user_indices]
    recommendations = set()
    for similar_user in similar_users_ids:
        recommended_games = df_merged[df_merged['user_id'] == similar_user]['title'].unique()
        recommendations.update(recommended_games)
    user_played_games = df_merged[df_merged['user_id'] == user_id]['title'].unique()
    recommendations = [game for game in recommendations if game not in user_played_games]
    return list(recommendations)[:top_n]

# --- Recommendation #9 - Popularity-Based Recommendation ---
def popularity_based_recommendation(df_games, df_users, num_recommendations=10, genre=None):
    """
    Recommends popular games based on total playtime, average playtime, and number of unique players.
    """
    if 'genres' in df_games.columns:
        df_merged = pd.merge(df_games[['app_id', 'title', 'genres']], df_users, on='app_id')
    else:
        df_merged = pd.merge(df_games[['app_id', 'title']], df_users, on='app_id')
    df_popularity = df_merged.groupby('title').agg(
        total_playtime=('total_playtime', 'sum'),
        avg_playtime=('total_playtime', 'mean'),
        num_players=('user_id', 'nunique'),
        first_genre=('genres', 'first') if 'genres' in df_games.columns else ('app_id', 'count') # Placeholder if no genres
    ).reset_index()
    if genre and 'first_genre' in df_popularity.columns:
        df_popularity = df_popularity[df_popularity['first_genre'].str.contains(genre, case=False, na=False)]
    df_popularity['popularity_score'] = (
        df_popularity['total_playtime'] * 0.5 +
        df_popularity['avg_playtime'] * 0.3 +
        df_popularity['num_players'] * 0.2
    )
    df_recommendations = df_popularity.sort_values('popularity_score', ascending=False).head(num_recommendations)
    return df_recommendations['title'].tolist()

# --- Recommendation #10 - Matrix Factorization Model ---
def build_matrix_factorization_model(df_users, num_factors=50, regularization=0.01, iterations=15, min_games_played=5):
    """
    Builds a matrix factorization model using the ALS algorithm from the implicit library.
    """
    df_filtered = df_users.groupby('user_id').filter(lambda x: len(x) >= min_games_played)
    user_item_matrix = pd.pivot_table(df_filtered, values='total_playtime', index='user_id', columns='app_id', fill_value=0)
    user_item_matrix_sparse = csr_matrix(user_item_matrix.values)
    model = implicit.als.AlternatingLeastSquares(factors=num_factors, regularization=regularization, iterations=iterations)
    model.fit(user_item_matrix_sparse)
    app_id_to_index = {app_id: index for index, app_id in enumerate(user_item_matrix.columns)}
    index_to_app_id = {index: app_id for app_id, index in app_id_to_index.items()}
    return model, app_id_to_index, index_to_app_id, user_item_matrix_sparse, user_item_matrix

def get_matrix_factorization_recommendations(user_id, model, user_item_matrix_sparse, index_to_app_id, df_games, user_item_matrix, top_n=10):
    """
    Generates recommendations for a given user using a trained matrix factorization model.
    """
    if user_id not in user_item_matrix.index:
        print(f"User ID '{user_id}' not found in the data.")
        return []
    user_row_index = user_item_matrix.index.get_loc(user_id)
    user_items = user_item_matrix_sparse[user_row_index]
    try:
        recommendations_indices, _ = model.recommend(user_id, user_items, N=top_n)
        recommendations_app_ids = [index_to_app_id[index] for index in recommendations_indices]
        recommendations_titles = df_games[df_games['app_id'].isin(recommendations_app_ids)]['title'].tolist()
        return recommendations_titles
    except Exception as e:
        print(f"Error during recommendation for user {user_id}: {e}")
        return []

if __name__ == "__main__":
    # Example usage (assuming you have loaded your dataframes as df_games and df_users)
    # You would need to replace this with your actual data loading and usage scenarios

    # Placeholder data (replace with your actual DataFrames)
    data = {'app_id': [1, 1, 2, 2, 3],
            'title': ['Game A', 'Game B', 'Game A', 'Game C', 'Game B'],
            'genres': ['Action', 'Adventure', 'Action', 'RPG', 'Adventure'],
            'rating': ['Positive', 'Very Positive', 'Positive', 'Mixed', 'Very Positive'],
            'rating_score': [4-6],
            'user_id': [1, 1, 2, 2, 3],
            'total_playtime': [7]}
    df_games_example = pd.DataFrame(data)
    df_users_example = pd.DataFrame(data[['app_id', 'user_id', 'total_playtime']])

    print("--- Content-Based Recommendation (Title) ---")
    print(recommendations_by_title(df_games_example, "title", "Game A"))

    print("\n--- Content-Based Recommendation (Genres and Rating) ---")
    print(content_based_recommendation_genres_rating("Game A", df_games_example))

    print("\n--- Content-Based Recommendation (Genres, Rating, Metacritic - Annoy) ---")
    df_games_example['metacritic_score'] = [4]
    print(content_based_recommendation_genres_rating_metacritic_annoy("Game A", df_games_example))

    print("\n--- Content-Based Recommendation (All Features - Annoy) ---")
    print(content_based_recommendation_all_features_annoy("Game A", df_games_example))

    print("\n--- Collaborative Filtering (Playtime) ---")
    print(collaborative_filtering_by_playtime(1, df_users_example))

    print("\n--- Collaborative Filtering (Rating and Playtime) ---")
    print(collaborative_filtering_with_ratings(1, df_games_example[['user_id', 'title', 'rating', 'total_playtime']].drop_duplicates()))

    print("\n--- Collaborative Filtering (Top 5 Playtime) ---")
    print(collaborative_filtering_top_5_playtime(1, df_games_example[['app_id', 'user_id', 'title', 'rating', 'total_playtime']].drop_duplicates()))

    print("\n--- Hybrid Recommendation (Top 10 Playtime) ---")
    print(hybrid_recommendation_top_10_playtime(1, df_games_example[['app_id', 'title', 'genres', 'rating']].drop_duplicates(), df_users_example))

    print("\n--- Popularity-Based Recommendation ---")
    print(popularity_based_recommendation(df_games_example, df_users_example))

    print("\n--- Matrix Factorization Recommendation ---")
    model_mf, app_id_to_index_mf, index_to_app_id_mf, user_item_matrix_sparse_mf, user_item_matrix_mf = build_matrix_factorization_model(df_users_example)
    if model_mf is not None:
        print(get_matrix_factorization_recommendations(1, model_mf, user_item_matrix_sparse_mf, index_to_app_id_mf, df_games_example, user_item_matrix_mf))
Key elements of this recommendation_models.py structure: