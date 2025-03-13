# main.py

# Import necessary libraries and modules
import pandas as pd
import os
# Assuming these files will be created based on the notebook's logic
# import data_preprocessing
# import training_orchestrator
# import recommendation_models
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from annoy import AnnoyIndex
import numpy as np
import implicit
from scipy.sparse import csr_matrix

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(data_paths):
    """Loads and preprocesses the datasets."""
    df_games_1 = pd.read_csv(os.path.join(data_paths['path_1'], 'games.csv')) if 'path_1' in data_paths and os.path.exists(os.path.join(data_paths['path_1'], 'games.csv')) else None
    df_games_2 = pd.read_csv(os.path.join(data_paths['path_2'], 'game_data_all.csv')) if 'path_2' in data_paths and os.path.exists(os.path.join(data_paths['path_2'], 'game_data_all.csv')) else None
    df_games_3 = pd.read_csv(os.path.join(data_paths['path_4'], 'Steam Games 2024.csv')) if 'path_4' in data_paths and os.path.exists(os.path.join(data_paths['path_4'], 'Steam Games 2024.csv')) else None
    df_users = pd.read_csv(os.path.join(data_paths['path_3'], 'user_game_played_data.csv')) if 'path_3' in data_paths and os.path.exists(os.path.join(data_paths['path_3'], 'user_game_played_data.csv')) else None
    df_recommendations = pd.read_csv(os.path.join(data_paths['path_1'], 'recommendations.csv')) if 'path_1' in data_paths and os.path.exists(os.path.join(data_paths['path_1'], 'recommendations.csv')) else None

    # --- Apply preprocessing steps from the notebook ---
    if df_games_1 is not None:
        df_games_1 = reduce_memory(df_games_1)
        columns_to_drop_df_games_1 = ['date_release', 'win', 'mac', 'linux', 'positive_ratio',  'user_reviews', 'price_final', 'price_original', 'discount', 'steam_deck']
        common_columns_df_games_1 = list(set(df_games_1.columns) & set(columns_to_drop_df_games_1))
        df_games_1.drop(columns=common_columns_df_games_1, axis=1, inplace=True)

    if df_games_2 is not None:
        df_games_2 = reduce_memory(df_games_2)
        columns_to_drop_df_games_2 = ['Unnamed: 0', 'release', 'peak_players', 'positive_reviews', 'negative_reviews', 'total_reviews', 'primary_genre', 'store_genres', 'publisher', 'developer', 'detected_technologies', 'store_asset_mod_time', 'review_percentage', 'players_right_now', '24_hour_peak', 'all_time_peak', 'all_time_peak_date']
        common_columns_df_games_2 = list(set(df_games_2.columns) & set(columns_to_drop_df_games_2))
        df_games_2.drop(columns=common_columns_df_games_2, axis=1, inplace=True)
        df_games_2.rename(columns={'game': 'title', 'link': 'app_id', 'rating': 'rating_score'}, inplace=True)
        df_games_2['app_id'] = df_games_2['app_id'].str.replace(r'/app/','', regex=True).replace(r'/','', regex=True).astype('int32')
        df_games_2 = df_games_2[['app_id', 'title', 'rating_score']]

    if df_games_3 is not None:
        df_games_3 = reduce_memory(df_games_3)
        columns_to_drop_df_games_3 = ['Release date','Estimated owners', 'Peak CCU', 'Required age', 'Price', 'Discount', 'DLC count', 'About the game', 'Supported languages', 'Full audio languages', 'Reviews', 'Header image', 'Website', 'Support url', 'Support email', 'Windows', 'Mac', 'Linux', 'Metacritic url', 'User score', 'Positive', 'Negative', 'Score rank', 'Achievements', 'Recommendations', 'Notes', 'Average playtime forever', 'Average playtime two weeks', 'Median playtime forever', 'Median playtime two weeks', 'Developers', 'Publishers', 'Categories', 'Tags', 'Screenshots', 'Movies']
        common_columns_df_games_3 = list(set(df_games_3.columns) & set(columns_to_drop_df_games_3))
        df_games_3.drop(columns=common_columns_df_games_3, axis=1, inplace=True)
        df_games_3.rename(columns={'AppID': 'app_id', 'Name': 'title', 'Metacritic score': 'metacritic_score', 'Genres': 'genres'}, inplace=True)
        df_games_3 = df_games_3[['app_id', 'title', 'genres', 'metacritic_score']]

    if df_users is not None:
        df_users = reduce_memory(df_users)
        df_users.rename(columns={'game_id': 'app_id', 'playtime_forever': 'total_playtime'}, inplace=True)
        df_users = df_users[['user_id', 'app_id', 'total_playtime']]
        df_users = df_users[df_users['total_playtime'] != 0]

    if df_recommendations is not None:
        df_recommendations = reduce_memory(df_recommendations)
        columns_to_drop_df_recommendations = ['helpful', 'funny', 'date', 'review_id']
        common_columns_df_recommendations = list(set(df_recommendations.columns) & set(columns_to_drop_df_recommendations))
        df_recommendations.drop(columns=common_columns_df_recommendations, axis=1, inplace=True)
        df_recommendations.rename(columns={'is_recommended': 'recommended', 'hours': 'total_hours'}, inplace=True)
        df_recommendations = df_recommendations[['user_id', 'app_id', 'recommended', 'total_hours']]

    # --- Merging DataFrames (as done in the notebook) ---
    df_games_f1 = pd.merge(df_games_1[['app_id', 'title', 'rating']] if df_games_1 is not None else pd.DataFrame(),
                             df_games_2[['app_id', 'title', 'rating_score']] if df_games_2 is not None else pd.DataFrame(),
                             on=['app_id', 'title'], how='inner') if df_games_1 is not None and df_games_2 is not None else None

    df_games = pd.merge(df_games_f1[['app_id', 'title', 'rating', 'rating_score']] if df_games_f1 is not None else pd.DataFrame(),
                        df_games_3[['app_id', 'title', 'genres', 'metacritic_score']] if df_games_3 is not None else pd.DataFrame(),
                        on=['app_id', 'title'], how='inner') if df_games_f1 is not None and df_games_3 is not None else None

    df_games_users_all = pd.merge(df_games[['app_id', 'title', 'rating', 'rating_score']] if df_games is not None else pd.DataFrame(),
                                  df_users, on='app_id') if df_games is not None and df_users is not None else None
    if df_games_users_all is not None:
        df_games_users_all = df_games_users_all[['user_id', 'title', 'rating', 'rating_score', 'total_playtime']]

    return df_games, df_users, df_games_users_all

def reduce_memory(df):
    """Reduces memory usage of a DataFrame."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
        if df[col].dtype == 'object' and col not in ['title', 'genres']:
            df[col] = df[col].astype('category')
    return df

# --- Recommendation Functions (These would ideally be in recommendation_models.py) ---
def recommendations(dataframe, title_col, game_name):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(dataframe[title_col].fillna('')) # Fill NaN in titles
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(dataframe.index, index=dataframe[title_col]).drop_duplicates()
    try:
        idx = indices[game_name]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[2], reverse=True)
        sim_scores = sim_scores[1:11]
        game_indices = [i for i in sim_scores]
        return dataframe['title'].iloc[game_indices]
    except KeyError:
        print(f"Game '{game_name}' not found.")
        return []

def content_based_recommendation_genre_rating(game_title, df, genre_weight=0.5, rating_score_weight=0.5, similarity_threshold=0.6):
    df['genres'] = df['genres'].astype(str).fillna('')
    tfidf_genres = TfidfVectorizer(stop_words='english')
    tfidf_matrix_genres = tfidf_genres.fit_transform(df['genres'])
    cosine_sim_genres = cosine_similarity(tfidf_matrix_genres, tfidf_matrix_genres)
    df['rating_score'] = pd.to_numeric(df['rating_score'], errors='coerce').fillna(0)
    df['rating_score_norm'] = (df['rating_score'] - df['rating_score'].min()) / (df['rating_score'].max() - df['rating_score'].min())
    similarity_matrix = (cosine_sim_genres * genre_weight) + (df['rating_score_norm'].values.reshape(-1, 1) * rating_score_weight)
    try:
        idx = df[df['title'] == game_title].index
        rated_games = {game_title}
        recommendations = []
        for i, score in enumerate(similarity_matrix[idx]):
            if i != idx and df['title'].iloc[i] not in rated_games and score > similarity_threshold:
                recommendations.append((df['title'].iloc[i], score))
                rated_games.add(df['title'].iloc[i])
        recommendations = sorted(recommendations, key=lambda x: x[2], reverse=True)[:10]
        return recommendations
    except IndexError:
        print(f"Game '{game_title}' not found.")
        return []

def collaborative_filtering_recommendation(user_id, df):
    user_item_matrix = pd.pivot_table(df, values='total_playtime', index='user_id', columns='title', fill_value=0)
    if user_id not in user_item_matrix.index:
        print(f"User ID '{user_id}' not found.")
        return []
    user_item_matrix = user_item_matrix.fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
    user_index = user_item_matrix.index.get_loc(user_id)
    user_similarity_scores = user_similarity[user_index]
    N = 5
    top_similar_users_indices = np.argsort(user_similarity_scores)[::-1][1:N+1]
    top_similar_users = user_item_matrix.index[top_similar_users_indices]
    recommendations = []
    for user in top_similar_users:
        played_games = df[df['user_id'] == user]['title'].unique()
        recommendations.extend(played_games)
    user_played_games = df[df['user_id'] == user_id]['title'].unique()
    recommendations = [game for game in set(recommendations) if game not in user_played_games]
    return recommendations[:20]

def build_matrix_factorization_model(df_users, df_games, num_factors=50, regularization=0.01, iterations=15):
    df_filtered = df_users.groupby('user_id').filter(lambda x: len(x) >= 5)
    user_item_matrix = pd.pivot_table(df_filtered, values='total_playtime', index='user_id', columns='app_id', fill_value=0)
    user_item_matrix_sparse = csr_matrix(user_item_matrix.values)
    model = implicit.als.AlternatingLeastSquares(factors=num_factors, regularization=regularization, iterations=iterations)
    model.fit(user_item_matrix_sparse)
    app_id_to_index = {app_id: index for index, app_id in enumerate(user_item_matrix.columns)}
    index_to_app_id = {index: app_id for app_id, index in app_id_to_index.items()}
    return model, app_id_to_index, index_to_app_id, user_item_matrix_sparse, user_item_matrix

def get_recommendations_mf(user_id, model, user_item_matrix_sparse, index_to_app_id, df_games, user_item_matrix, N=10):
    if user_id not in user_item_matrix.index:
        print(f"User ID '{user_id}' not found.")
        return []
    user_row_index = user_item_matrix.index.get_loc(user_id)
    user_items = user_item_matrix_sparse[user_row_index]
    try:
        recommendations_indices, _ = model.recommend(user_id, user_items, N=N)
        recommendations_app_ids = [index_to_app_id[index] for index in recommendations_indices]
        recommendations_titles = df_games[df_games['app_id'].isin(recommendations_app_ids)]['title'].tolist()
        return recommendations_titles
    except KeyError:
        print(f"Could not generate recommendations for user {user_id}.")
        return []

# --- Main Execution Block ---
if __name__ == "__main__":
    # Define paths to your datasets - ADJUST THESE TO YOUR LOCAL SETUP
    data_paths = {
        "path_1": "./data/antonkozyriev_game-recommendations-on-steam",
        "path_2": "./data/whigmalwhim_steam-releases",
        "path_3": "./data/maestrocor_steam-games-and-user-playtime-data",
        "path_4": "./data/praffulsingh009_steam-video-games-2024",
    }

    # Load and preprocess data
    df_games, df_users, df_games_users_all = load_and_preprocess_data(data_paths)

    if df_games is not None and df_users is not None and df_games_users_all is not None:
        # --- Example Usage of Recommendation Models ---

        print("\n--- Content-Based Recommendation based on Title ---")
        if 'title' in df_games.columns:
            game_to_find = "Grand Theft Auto V"
            title_recommendations = recommendations(df_games.copy(), "title", game_to_find)
            if title_recommendations is not None and not title_recommendations.empty:
                print(f"Recommendations for '{game_to_find}':")
                for i, rec in enumerate(title_recommendations, 1):
                    print(f"{i}. {rec}")
            else:
                print(f"Could not find recommendations for '{game_to_find}'.")
        else:
            print("Column 'title' not found in df_games for title-based recommendation.")

        print("\n--- Content-Based Recommendation based on Genre and Rating Score ---")
        if 'title' in df_games.columns and 'genres' in df_games.columns and 'rating_score' in df_games.columns:
            game_to_find_genre_rating = "Counter-Strike"
            genre_rating_recommendations = content_based_recommendation_genre_rating(game_to_find_genre_rating, df_games.copy())
            if genre_rating_recommendations:
                print(f"Recommendations for '{game_to_find_genre_rating}':")
                for title, score in genre_rating_recommendations:
                    print(f"- {title} (Similarity Score: {score:.2f})")
            else:
                print(f"Could not find recommendations for '{game_to_find_genre_rating}'.")
        else:
            print("One or more required columns ('title', 'genres', 'rating_score') not found in df_games for genre-rating based recommendation.")

        print("\n--- Collaborative Filtering Recommendation ---")
        if 'user_id' in df_games_users_all.columns:
            user_id_to_recommend = 10
            collaborative_recs = collaborative_filtering_recommendation(user_id_to_recommend, df_games_users_all.copy())
            if collaborative_recs:
                print(f"Collaborative Filtering Recommendations for user {user_id_to_recommend}:")
                for i, rec in enumerate(collaborative_recs, 1):
                    print(f"{i}. {rec}")
            else:
                print(f"Could not find collaborative filtering recommendations for user {user_id_to_recommend}.")
        else:
            print("Column 'user_id' not found in df_games_users_all for collaborative filtering.")

        print("\n--- Matrix Factorization Recommendation ---")
        if 'user_id' in df_users.columns and 'app_id' in df_users.columns and 'title' in df_games.columns:
            try:
                mf_model, app_id_to_index, index_to_app_id, user_item_matrix_sparse, user_item_matrix = build_matrix_factorization_model(df_users.copy(), df_games.copy())
                user_id_mf = 15
                mf_recommendations = get_recommendations_mf(user_id_mf, mf_model, user_item_matrix_sparse, index_to_app_id, df_games.copy(), user_item_matrix)
                if mf_recommendations:
                    print(f"Matrix Factorization Recommendations for user {user_id_mf}:")
                    for i, rec in enumerate(mf_recommendations, 1):
                        print(f"{i}. {rec}")
                else:
                    print(f"Could not find matrix factorization recommendations for user {user_id_mf}.")
            except Exception as e:
                print(f"Error during Matrix Factorization: {e}")
        else:
            print("One or more required columns ('user_id', 'app_id', 'title') not found for matrix factorization.")

    else:
        print("Error: Could not load and preprocess data. Please check the data paths.")