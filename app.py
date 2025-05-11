from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# --- Load Data and Models ---
DATA_FILE = '/Users/rsusanto/Documents/Flask Local/data/df_games.csv'
TFIDF_FILE_1 = '/Users/rsusanto/Documents/Flask Local/models/tfidf_vectorizer_rec1.pkl'
TFIDF_FILE_2 = '/Users/rsusanto/Documents/Flask Local/models/tfidf_vectorizer_rec2.pkl'

df_games = pd.DataFrame()
tfidf_vectorizer_1 = None
tfidf_vectorizer_2 = None

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(script_dir, DATA_FILE)
    tfidf_file_path_1 = os.path.join(script_dir, TFIDF_FILE_1)
    tfidf_file_path_2 = os.path.join(script_dir, TFIDF_FILE_2)

    df_games = pd.read_csv(data_file_path)

    with open(tfidf_file_path_1, 'rb') as f:
        tfidf_vectorizer_1 = pickle.load(f)

    with open(tfidf_file_path_2, 'rb') as f:
        tfidf_vectorizer_2 = pickle.load(f)

except FileNotFoundError as e:
    print(f"Error: File not found - {e.filename}. Please check the file path.")
except Exception as e:
    print(f"Error loading data or model: {e}")


# --- Recommendation Functions ---
def get_recommendations_title(dataframe, title_col, game_name, tfidf_vectorizer):
    """Recommends games based on title similarity."""

    if tfidf_vectorizer is None or dataframe.empty:
        return pd.Series(["Error: Data or model not loaded."])

    if game_name not in dataframe[title_col].values:
        return pd.Series([f"Game '{game_name}' not found in the database."])

    try:
        game_name_tfidf = tfidf_vectorizer.transform([game_name])
        tfidf_matrix = tfidf_vectorizer.transform(dataframe[title_col])
        cosine_sim = cosine_similarity(game_name_tfidf, tfidf_matrix)
        indices = pd.Series(dataframe.index, index=dataframe[title_col]).drop_duplicates()
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Top 10 excluding the input game
        game_indices = [i[0] for i in sim_scores]
        return dataframe['title'].iloc[game_indices]
    except KeyError:
        return pd.Series([f"Game '{game_name}' not found in the indices."])
    except Exception as e:
        return pd.Series([f"Error during recommendation: {e}"])


def get_recommendations_combined(dataframe, title_col, rating_col, game_name, tfidf_vectorizer, rating_weight=0.2):
    """Recommends games based on title similarity and rating."""

    if tfidf_vectorizer is None or dataframe.empty:
        return pd.Series(["Error: Data or model not loaded."])

    if game_name not in dataframe[title_col].values:
        return pd.Series([f"Game '{game_name}' not found in the database."])

    try:
        indices = pd.Series(dataframe.index, index=dataframe[title_col]).drop_duplicates()
        idx = indices[game_name]

        game_name_tfidf = tfidf_vectorizer.transform([game_name])
        tfidf_matrix = tfidf_vectorizer.transform(dataframe[title_col])
        cosine_sim = cosine_similarity(game_name_tfidf, tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim[0]))

        base_rating = dataframe.loc[idx, rating_col]

        combined_scores = []
        for i, score in sim_scores:
            other_rating = dataframe.loc[i, rating_col]

            min_rating = dataframe[rating_col].min()
            max_rating = dataframe[rating_col].max()
            normalized_base_rating = (base_rating - min_rating) / (max_rating - min_rating)
            normalized_other_rating = (other_rating - min_rating) / (max_rating - min_rating)

            combined_score = (1 - rating_weight) * score + rating_weight * normalized_other_rating
            combined_scores.append((i, combined_score))

        combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)
        combined_scores = [s for s in combined_scores if s[0] != idx]
        top_10_indices = [i[0] for i in combined_scores[:10]]

        return dataframe['title'].iloc[top_10_indices]

    except KeyError:
        return pd.Series([f"Game '{game_name}' not found in the indices."])
    except Exception as e:
        return pd.Series([f"Error during recommendation: {e}"])


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    game_name = request.form['game_title']

    recommendations_title = get_recommendations_title(
        df_games, "title", game_name, tfidf_vectorizer_1
    )
    recommendations_combined = get_recommendations_combined(
        df_games, "title", "rating_score", game_name, tfidf_vectorizer_2, rating_weight=0.8
    )

    if isinstance(recommendations_title, pd.Series):
        recommendations_title_list = recommendations_title.tolist()
    else:
        recommendations_title_list = list(recommendations_title)

    if isinstance(recommendations_combined, pd.Series):
        recommendations_combined_list = recommendations_combined.tolist()
    else:
        recommendations_combined_list = list(recommendations_combined)

    return render_template(
        'recommendations.html',
        game_name=game_name,
        recommendations_title=recommendations_title_list,
        recommendations_combined=recommendations_combined_list,
    )


if __name__ == '__main__':
    app.run(debug=True)