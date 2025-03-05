import streamlit as st
   import pickle
   from model_training import *  # Import functions from model_training.py

   # Load the trained model
   model = pickle.load(open('model.pkl', 'rb'))  

   # Streamlit UI elements
   st.title('UCSD Video Games Recommender')
   game_name = st.text_input('Enter a game name:')

   if st.button('Recommend'):
       recommendations = get_recommendations(model, game_name)  # Call recommendation function
       st.write(recommendations)

   # ... (Add other UI elements and logic as needed) ...
