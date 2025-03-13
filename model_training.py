# model_training.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import kagglehub  # If you need to download data directly from Kaggle

# --- 1. Download Data (if needed) ---
dataset_name = "antonkozyriev/game-recommendations-on-steam"
output_path = "data"
if not os.path.exists(output_path):
    os.makedirs(output_path)
try:
    kagglehub.dataset_download(dataset_name, path=output_path, force=False, quiet=False)
except Exception as e:
    print(f"Error downloading dataset: {e}. Make sure you have kagglehub configured.")

# --- 2. Load and Preprocess Data ---
games_df_path = os.path.join(output_path, 'games.csv')
try:
    df_games = pd.read_csv(games_df_path)
    print(f"Successfully loaded data from: {games_df_path}")
except FileNotFoundError:
    print(f"Error: The file '{games_df_path}' was not found.")
    exit()

# --- 3. Train the Recommendation Model (Content-Based on Title) ---
# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words="english")

# Fit and transform the game titles
tfidf_matrix = tfidf.fit_transform(df_games['title'])

# Calculate the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create indices of game names
indices = pd.Series(df_games.index, index=df_games['title']).drop_duplicates()

# --- 4. Save the Trained Model Components ---
model_dir = "trained_model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

tfidf_path = os.path.join(model_dir, 'tfidf_model.pkl')
similarity_path = os.path.join(model_dir, 'cosine_similarity.pkl')
indices_path = os.path.join(model_dir, 'indices.pkl')

try:
    with open(tfidf_path, 'wb') as f:
        pickle.dump(tfidf, f)
    print(f"TF-IDF model saved to: {tfidf_path}")

    with open(similarity_path, 'wb') as f:
        pickle.dump(cosine_sim, f)
    print(f"Cosine similarity matrix saved to: {similarity_path}")

    with open(indices_path, 'wb') as f:
        pickle.dump(indices, f)
    print(f"Game title indices saved to: {indices_path}")

    print("**The 'brain' of your game recommendation toy (TF-IDF model, similarity matrix, and indices) has been created and saved!**")

except Exception as e:
    print(f"Error saving model components: {e}")

if __name__ == "__main__":
    # You can add code here to test your training script
    print("\nTraining script executed successfully. Model components saved in 'trained_model' directory.")