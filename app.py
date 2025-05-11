from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# --- Load your game data and the combined recommendation model (TF-IDF vectorizer) ---
# Replace with your actual file paths
DATA_FILE = '/Users/rsusanto/Documents/Flask Local/data/df_games.csv'
MODEL_FILE = '/Users/rsusanto/Documents/Flask Local/models/tfidf_vectorizer_rec2.pkl' # Assuming your saved model includes the TF-IDF

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(script_dir, DATA_FILE)
    model_file_path = os.path.join(script_dir, MODEL_FILE)
    df_games = pd.read_csv(data_file_path)
    with open(model_file_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: File not found - {e.filename}. Please check the file path.")
    df_games = pd.DataFrame()
    tfidf_vectorizer = None
except Exception as e:
    print(f"Error loading data or model: {e}")
    df_games = pd.DataFrame()
    tfidf_vectorizer = None

# --- Recommendation Function (Now Handles Title and Rating) ---
def get_combined_recommendations(dataframe, title_col, rating_col, game_name, tfidf_vectorizer, rating_weight=0.2):
    """
    Generates recommendations based on title similarity and rating score.

    Args:
        dataframe (pd.DataFrame): The input dataframe containing game information.
        title_col (str): The name of the column containing game titles.
        rating_col (str): The name of the column containing game rating scores.
        game_name (str): The name of the game to find recommendations for.
        tfidf_vectorizer (TfidfVectorizer): The pre-fitted TF-IDF vectorizer.
        rating_weight (float, optional): The weight to give to the rating score (between 0 and 1). Defaults to 0.2.

    Returns:
        pd.Series: A Series containing the titles of the top 10 recommended games,
                  or an empty Series if an error occurs.
    """
    if tfidf_vectorizer is None or dataframe.empty:
        return pd.Series(["Error: Data or model not loaded."])

    if game_name not in dataframe[title_col].values:
        return pd.Series([f"Game '{game_name}' not found in the database."])

    try:
        indices = pd.Series(dataframe.index, index=dataframe[title_col]).drop_duplicates()
        idx = indices[game_name]

        # Get the similarity scores based on title
        game_name_tfidf = tfidf_vectorizer.transform([game_name])
        tfidf_matrix = tfidf_vectorizer.transform(dataframe[title_col])
        cosine_sim = cosine_similarity(game_name_tfidf, tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim[0]))

        # Get the rating score of the input game
        base_rating = dataframe.loc[idx, rating_col]

        # Create a list to store combined scores
        combined_scores = []
        for i, score in sim_scores:
            # Get the rating of the other game
            other_rating = dataframe.loc[i, rating_col]

            # Normalize ratings
            min_rating = dataframe[rating_col].min()
            max_rating = dataframe[rating_col].max()
            normalized_base_rating = (base_rating - min_rating) / (max_rating - min_rating)
            normalized_other_rating = (other_rating - min_rating) / (max_rating - min_rating)

            # Calculate the weighted average of similarity and normalized rating
            combined_score = (1 - rating_weight) * score + rating_weight * normalized_other_rating
            combined_scores.append((i, combined_score))

        # Sort the combined scores
        combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)

        # Get the top 10 most similar games (excluding the input game itself)
        combined_scores = [s for s in combined_scores if s[0] != idx] # Exclude the input game
        top_10_indices = [i[0] for i in combined_scores[:10]]

        return dataframe['title'].iloc[top_10_indices]

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
    """Handles the form submission and displays recommendations based on title and rating."""
    game_name = request.form['game_title']
    recommendations_list = get_combined_recommendations(df_games, "title", "rating_score", game_name, tfidf_vectorizer, rating_weight=0.8) # Assuming 'rating_score' is your rating column

    if isinstance(recommendations_list, pd.Series):
        recommendations = recommendations_list.tolist()
    else:
        recommendations = list(recommendations_list)

    return render_template('recommendations.html', game_name=game_name, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)