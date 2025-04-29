from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from google.cloud import storage  # Import Google Cloud Storage library
import tempfile  # For creating temporary files

app = Flask(__name__)

# --- Google Cloud Storage Configuration ---
BUCKET_NAME = "capstone-rs"  # Replace with your bucket name
DATA_FILE_BLOB_NAME = "data/df_games.csv"  # Path to your CSV in the bucket
TFIDF_FILE_BLOB_NAME = "models/tfidf_vectorizer_rec1.pkl"  # Path to your model

def download_from_gcs(bucket_name, blob_name, local_file_path):
    """Downloads a file from Google Cloud Storage."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_file_path)
        print(f"Downloaded {blob_name} to {local_file_path}")
    except Exception as e:
        print(f"Error downloading {blob_name}: {e}")
        return False
    return True

# --- Load data and model ---
df_games = pd.DataFrame()  # Initialize as empty
tfidf_vectorizer = None

try:
    # Create temporary files to store the downloaded data
    with tempfile.NamedTemporaryFile(delete=False) as tmp_csv:
        local_csv_path = tmp_csv.name
    with tempfile.NamedTemporaryFile(delete=False) as tmp_pkl:
        local_pkl_path = tmp_pkl.name

    # Download from GCS
    if (download_from_gcs(BUCKET_NAME, DATA_FILE_BLOB_NAME, local_csv_path) and
        download_from_gcs(BUCKET_NAME, TFIDF_FILE_BLOB_NAME, local_pkl_path)):

        df_games = pd.read_csv(local_csv_path)
        with open(local_pkl_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        # Clean up temporary files (important!)
        os.remove(local_csv_path)
        os.remove(local_pkl_path)

    else:
        print("Failed to download data/model from GCS. App may not function correctly.")

except Exception as e:
    print(f"Error loading data: {e}")

# --- Recommendation Function ---
def get_recommendations(dataframe, title_col, game_name, tfidf_vectorizer):
    """
    Recommends similar games based on title using a pre-fitted TF-IDF vectorizer.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing game data.
        title_col (str): The name of the column containing game titles.
        game_name (str): The name of the game to find recommendations for.
        tfidf_vectorizer (TfidfVectorizer): The pre-fitted TF-TFIDF vectorizer.

    Returns:
        pd.Series: A Series containing the titles of the top 10 recommended games,
                  or an empty Series if an error occurs.
    """
    if tfidf_vectorizer is None or dataframe.empty:
        return pd.Series(["Error: Data or model not loaded."])

    if game_name not in dataframe[title_col].values:
        return pd.Series([f"Game '{game_name}' not found in the database."])

    try:
        #  Transform the input game name into a TF-TFIDF vector
        game_name_tfidf = tfidf_vectorizer.transform([game_name])  

        tfidf_matrix = tfidf_vectorizer.transform(dataframe[title_col])
        cosine_sim = cosine_similarity(game_name_tfidf, tfidf_matrix) # Calculate cosine similarity between the input game and all games in the dataframe
        indices = pd.Series(dataframe.index, index=dataframe[title_col]).drop_duplicates()
        # idx = indices[game_name] # No longer used
        sim_scores = list(enumerate(cosine_sim[0]))  #  cosine_similarity returns a 2D array, get the first row
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Top 10 excluding the input game
        game_indices = [i[0] for i in sim_scores]
        return dataframe['title'].iloc[game_indices]
    except KeyError:
        return pd.Series([f"Game '{game_name}' not found in the indices."])
    except Exception as e:
        return pd.Series([f"Error during recommendation: {e}"])

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the input form."""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """Handles the form submission and displays recommendations."""
    game_name = request.form['game_title']
    recommendations_list = get_recommendations(df_games, "title", game_name, tfidf_vectorizer)

    # Convert recommendations to a list of strings, handling potential errors
    if isinstance(recommendations_list, pd.Series):
        recommendations = recommendations_list.tolist()
    else:
        recommendations = list(recommendations_list)  # Ensure it is a list

    return render_template('recommendations.html', game_name=game_name, recommendations=recommendations)

#if __name__ == '__main__':
#    app.run(debug=True)