# data_preprocessing.py

import pandas as pd
import os
from kagglehub import dataset_download

def download_datasets(download_path="data"):
    """
    Downloads the necessary datasets from Kaggle if they don't exist locally.
    """
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    datasets = {
        "antonkozyriev/game-recommendations-on-steam": "Size M",
        "whigmalwhim/steam-releases": "Size L",
        "maestrocor/steam-games-and-user-playtime-data": "Size XL (users)",
        "praffulsingh009/steam-video-games-2024": "Size XL (games)"
        # Note: Datasets in [1] are listed as not being used after multiple iterations.
    }

    paths = {}
    for dataset_name, description in datasets.items():
        dataset_path = os.path.join(download_path, dataset_name.split('/')[2])
        if not os.path.exists(dataset_path):
            print(f"Downloading {description} dataset: {dataset_name}...")
            try:
                paths[dataset_name.split('/')[2]] = dataset_download(dataset_name, path=download_path, force=False, quiet=False)
            except Exception as e:
                print(f"Error downloading {dataset_name}: {e}")
                paths[dataset_name.split('/')[2]] = None
        else:
            print(f"{description} dataset: {dataset_name} already exists.")
            paths[dataset_name.split('/')[2]] = dataset_path
    return paths

def reduce_memory(df):
    """
    Reduces the memory usage of a DataFrame by downcasting numerical columns.
    """
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
        if df[col].dtype == 'object' and col not in ['title', 'genres']:
            df[col] = df[col].astype('category')
    return df

def load_and_process_data(data_paths):
    """
    Loads the datasets into pandas DataFrames, reduces memory usage,
    removes unnecessary columns, renames columns, and merges relevant DataFrames.
    """
    df_games_1, df_games_2, df_games_3, df_users, df_recommendations = None, None, None, None, None

    if data_paths.get('game-recommendations-on-steam'):
        df_games_1_path = os.path.join(data_paths['game-recommendations-on-steam'], 'games.csv')
        df_recommendations_path = os.path.join(data_paths['game-recommendations-on-steam'], 'recommendations.csv')
        df_games_1 = reduce_memory(pd.read_csv(df_games_1_path))
        df_recommendations = reduce_memory(pd.read_csv(df_recommendations_path))

    if data_paths.get('steam-releases'):
        df_games_2_path = os.path.join(data_paths['steam-releases'], 'game_data_all.csv')
        df_games_2 = reduce_memory(pd.read_csv(df_games_2_path))

    if data_paths.get('steam-games-and-user-playtime-data'):
        df_users_path = os.path.join(data_paths['steam-games-and-user-playtime-data'], 'user_game_played_data.csv')
        df_users = reduce_memory(pd.read_csv(df_users_path))

    if data_paths.get('steam-video-games-2024'):
        df_games_3_path = os.path.join(data_paths['steam-video-games-2024'], 'Steam Games 2024.csv')
        df_games_3 = reduce_memory(pd.read_csv(df_games_3_path))

    # --- Feature Engineering ---

    # Columns to drop [3]
    columns_to_drop_df_games_1 = ['date_release', 'win', 'mac', 'linux', 'positive_ratio', 'user_reviews', 'price_final', 'price_original', 'discount', 'steam_deck']
    columns_to_drop_df_games_2 = ['Unnamed: 0', 'release', 'peak_players', 'positive_reviews', 'negative_reviews', 'total_reviews', 'primary_genre', 'store_genres', 'publisher', 'developer', 'detected_technologies', 'store_asset_mod_time', 'review_percentage', 'players_right_now', '24_hour_peak', 'all_time_peak', 'all_time_peak_date']
    columns_to_drop_df_games_3 = ['Release date','Estimated owners', 'Peak CCU', 'Required age', 'Price', 'Discount', 'DLC count', 'About the game', 'Supported languages', 'Full audio languages', 'Reviews', 'Header image', 'Website', 'Support url', 'Support email', 'Windows', 'Mac', 'Linux', 'Metacritic url', 'User score', 'Positive', 'Negative', 'Score rank', 'Achievements', 'Recommendations', 'Notes', 'Average playtime forever', 'Average playtime two weeks', 'Median playtime forever', 'Median playtime two weeks', 'Developers', 'Publishers', 'Categories', 'Tags', 'Screenshots', 'Movies']
    columns_to_drop_df_recommendations = ['helpful', 'funny', 'date', 'review_id']

    # Drop common columns [4]
    if df_games_1 is not None:
        common_cols = list(set(df_games_1.columns) & set(columns_to_drop_df_games_1))
        df_games_1.drop(columns=common_cols, axis=1, inplace=True, errors='ignore')
    if df_games_2 is not None:
        common_cols = list(set(df_games_2.columns) & set(columns_to_drop_df_games_2))
        df_games_2.drop(columns=common_cols, axis=1, inplace=True, errors='ignore')
    if df_games_3 is not None:
        common_cols = list(set(df_games_3.columns) & set(columns_to_drop_df_games_3))
        df_games_3.drop(columns=common_cols, axis=1, inplace=True, errors='ignore')
    if df_recommendations is not None:
        common_cols = list(set(df_recommendations.columns) & set(columns_to_drop_df_recommendations))
        df_recommendations.drop(columns=common_cols, axis=1, inplace=True, errors='ignore')

    # Rename columns [5]
    if df_games_2 is not None:
        df_games_2.rename(columns={'game': 'title', 'link': 'app_id', 'rating': 'rating_score'}, inplace=True)
    if df_games_3 is not None:
        df_games_3.rename(columns={'AppID': 'app_id', 'Name': 'title', 'Metacritic score': 'metacritic_score', 'Genres': 'genres'}, inplace=True)
    if df_users is not None:
        df_users.rename(columns={'game_id': 'app_id', 'playtime_forever': 'total_playtime'}, inplace=True)
    if df_recommendations is not None:
        df_recommendations.rename(columns={'is_recommended': 'recommended', 'hours': 'total_hours'}, inplace=True)

    # Reorder columns [6] and clean app_id
    if df_games_2 is not None:
        df_games_2['app_id'] = df_games_2['app_id'].str.replace(r'/app/','', regex=True).replace(r'/','', regex=True).astype('int32', errors='ignore')
        df_games_2 = df_games_2[['app_id', 'title', 'rating_score']]
    if df_games_3 is not None:
        df_games_3 = df_games_3[['app_id', 'title', 'genres', 'metacritic_score']]
    if df_users is not None:
        df_users = df_users[['user_id', 'app_id', 'total_playtime']]
    if df_recommendations is not None:
        df_recommendations = df_recommendations[['user_id', 'app_id', 'recommended', 'total_hours']]

    # Merge DataFrames [7]
    df_games = None
    if df_games_1 is not None and df_games_2 is not None:
        df_games_f1 = pd.merge(
            df_games_1[['app_id', 'title', 'rating']],
            df_games_2[['app_id', 'title', 'rating_score']],
            on=['app_id', 'title'],
            how='inner'
        )
        if df_games_3 is not None:
            df_games = pd.merge(
                df_games_f1[['app_id', 'title', 'rating', 'rating_score']],
                df_games_3[['app_id', 'title', 'genres', 'metacritic_score']],
                on=['app_id', 'title'],
                how='inner'
            )
        else:
            df_games = df_games_f1

    # Handle potential missing df_games if merge conditions are not met
    if df_games is None and df_games_3 is not None:
        df_games = df_games_3[['app_id', 'title', 'genres', 'metacritic_score']]
    elif df_games is None and df_games_2 is not None:
        df_games = df_games_2[['app_id', 'title', 'rating_score']]
    elif df_games is None and df_games_1 is not None:
        df_games = df_games_1[['app_id', 'title', 'rating']]

    # Remove users with zero playtime [8]
    if df_users is not None:
        df_users = df_users[df_users['total_playtime'] != 0].copy()

    # Merge games and users [8]
    df_games_users_all = None
    if df_games is not None and df_users is not None:
        df_games_users_all = pd.merge(df_games[['app_id', 'title', 'rating', 'rating_score']], df_users, on='app_id').copy()
        df_games_users_all = df_games_users_all[['user_id', 'title', 'rating', 'rating_score', 'total_playtime']]

    # Convert 'metacritic_score' to numeric, handling errors [9]
    if df_games is not None:
        df_games['metacritic_score'] = pd.to_numeric(df_games['metacritic_score'], errors='coerce')

    return df_games, df_users, df_recommendations, df_games_users_all

if __name__ == "__main__":
    data_paths = download_datasets()
    if any(data_paths.values()):
        df_games, df_users, df_recommendations, df_games_users_all = load_and_process_data(data_paths)
        print("\nProcessed DataFrames:")
        if df_games is not None:
            print("df_games head:")
            print(df_games.head())
            print("df_games shape:", df_games.shape)
        if df_users is not None:
            print("\ndf_users head:")
            print(df_users.head())
            print("df_users shape:", df_users.shape)
        if df_recommendations is not None:
            print("\ndf_recommendations head:")
            print(df_recommendations.head())
            print("df_recommendations shape:", df_recommendations.shape)
        if df_games_users_all is not None:
            print("\ndf_games_users_all head:")
            print(df_games_users_all.head())
            print("df_games_users_all shape:", df_games_users_all.shape)
    else:
        print("No datasets were successfully downloaded.")