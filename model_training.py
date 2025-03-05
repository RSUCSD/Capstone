# model_training.py
import pandas as pd
import numpy as np
import os
import kagglehub
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import pickle

# Function to reduce the memory usage of a DataFrame.
def reduce_memory(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
        if df[col].dtype == 'object':
            if col not in ['title', 'genres']:
                df[col] = df[col].astype('category')
    return df

def load_data():
    # Download CSV from Kaggle
    path_1 = kagglehub.dataset_download("antonkozyriev/game-recommendations-on-steam")
    path_2 = kagglehub.dataset_download("whigmalwhim/steam-releases")
    path_4 = kagglehub.dataset_download("praffulsingh009/steam-video-games-2024")
    path_3 = kagglehub.dataset_download("maestrocor/steam-games-and-user-playtime-data")
    # Read the CSV files with the updated paths
    df_games_1 = reduce_memory(pd.read_csv(os.path.join(path_1, 'games.csv')))
    df_games_2 = reduce_memory(pd.read_csv(os.path.join(path_2, 'game_data_all.csv')))
    df_games_3 = reduce_memory(pd.read_csv(os.path.join(path_4, 'Steam Games 2024.csv')))
    df_users = reduce_memory(pd.read_csv(os.path.join(path_3, 'user_game_played_data.csv')))

    # ... (Rest of your data loading and preprocessing code from the Colab notebook) ...

    return df_games, df_games_users_all

def train_content_based_model(df_games):
    # Content-Based Recommendation based on genres, rating, rating score, and metacritic score with Annoy
    df_games['genres'] = df_games['genres'].astype(str)
    tfidf_genres = TfidfVectorizer(stop_words='english')
    tfidf_matrix_genres = tfidf_genres.fit_transform(df_games['genres'])
    df_games['rating_score'] = pd.to_numeric(df_games['rating_score'], errors='coerce')
    df_games['metacritic_score'] = pd.to_numeric(df_games['metacritic_score'], errors='coerce')
    df_games['rating_score_norm'] = (df_games['rating_score'] - df_games['rating_score'].min()) / (df_games['rating_score'].max() - df_games['rating_score'].min())
    df_games['metacritic_score_norm'] = (df_games['metacritic_score'] - df_games['metacritic_score'].min()) / (df_games['metacritic_score'].max() - df_games['metacritic_score'].min())
    le = LabelEncoder()
    df_games['rating_encoded'] = le.fit_transform(df_games['rating'])
    df_games['rating_encoded_norm'] = (df_games['rating_encoded'] - df_games['rating_encoded'].min()) / (df_games['rating_encoded'].max() - df_games['rating_encoded'].min())
    genre_weight = 0.25
    rating_weight = 0.25
    rating_score_weight = 0.25
    metacritic_score_weight = 0.25
    combined_features = np.hstack([
        tfidf_matrix_genres.toarray() * genre_weight,
        df_games['rating_encoded_norm'].values.reshape(-1, 1) * rating_weight,
        df_games['rating_score_norm'].values.reshape(-1, 1) * rating_score_weight,
        df_games['metacritic_score_norm'].values.reshape(-1, 1) * metacritic_score_weight
    ])
    f = combined_features.shape[1]
    t = AnnoyIndex(f, 'angular')
    for i in range(len(combined_features)):
        t.add_item(i, combined_features[i])
    t.build(10)
    return t  # Return the AnnoyIndex object

def save_model(model, filename):
    import pickle
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    df_games, df_games_users_all = load_data()
    content_based_model = train_content_based_model(df_games)
    save_model(content_based_model, 'content_based_model.ann')  # Save as .ann for AnnoyIndex
    save_model(df_games, 'df_games.pkl')  # Save df_games for later use in app.py
