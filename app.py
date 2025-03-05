# app.py
import streamlit as st
import pickle
from annoy import AnnoyIndex  # Import AnnoyIndex

# Load the trained model and data
content_based_model = AnnoyIndex(476, 'angular')  # Adjust dimensions as needed
content_based_model.load('content_based_model.ann')  # Load the .ann file

with open('df_games.pkl', 'rb') as file:
    df_games = pickle.load(file)

# Streamlit UI elements
st.title('UCSD Video Games Recommender')
game_name = st.text_input('Enter a game name:')

# Recommendation function
def get_recommendations(model, game_name, df_games=df_games, top_n=10):
    index = df_games[df_games['title'] == game_name].index[0] if game_name in df_games['title'].values else None
    if index is not None:
        similar_items = model.get_nns_by_item(index, top_n + 1, include_distances=True)
        recommendations = [df_games['title'].iloc[i] for i, dist in zip(similar_items[0], similar_items[1]) if i != index]
        return recommendations
    else:
        return ["Game not found in the dataset"]

if st.button('Recommend'):
    recommendations = get_recommendations(content_based_model, game_name)  # Pass the AnnoyIndex
    st.write(recommendations)
