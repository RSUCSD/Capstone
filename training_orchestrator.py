# training_orchestrator.py

import pandas as pd
# Import any necessary libraries for model training, e.g., scikit-learn, implicit
# from sklearn.model_selection import train_test_split # Example

# Assume recommendation_models.py contains the definitions for our recommendation models
from recommendation_models import (
    ContentBasedTitleRecommender,
    ContentBasedGenresRatingRecommender,
    ContentBasedAnnoyRecommender,
    CollaborativeFilteringPlaytimeRecommender,
    CollaborativeFilteringMultiCategoryRecommender,
    MatrixFactorizationRecommender,
    # ... other models
    save_model,
    load_model
)

class RecommendationTrainer:
    def __init__(self, data_path=None, model_output_path='trained_models'):
        """
        Initializes the RecommendationTrainer.

        Args:
            data_path (str, optional): Path to the processed data files. Defaults to None.
            model_output_path (str, optional): Path to save trained models. Defaults to 'trained_models'.
        """
        self.data_path = data_path
        self.model_output_path = model_output_path

    def load_processed_data(self):
        """
        Loads the processed DataFrames required for training.
        This would typically load the outputs from data_preprocessing.py.
        """
        # Placeholder for loading data
        print("Loading processed data...")
        try:
            self.df_games = pd.read_csv(f"{self.data_path}/processed_games.csv") # Example
            self.df_users = pd.read_csv(f"{self.data_path}/processed_users.csv") # Example
            self.df_games_users_all = pd.read_csv(f"{self.data_path}/processed_game_user_interactions.csv") # Example
            print("Data loaded successfully.")
        except FileNotFoundError:
            print("Error: Processed data files not found. Ensure data_preprocessing.py has been run.")
            self.df_games = None
            self.df_users = None
            self.df_games_users_all = None

    def train_content_based_title(self, output_filename='content_based_title_model.joblib'):
        """
        Trains the content-based filtering model based on game titles.
        (Corresponds to Recommendation #1 in the notebook)
        """
        if self.df_games is not None:
            print("Training Content-Based (Title) model...")
            model = ContentBasedTitleRecommender() # Assuming this class exists in recommendation_models.py
            model.fit(self.df_games, 'title')
            filepath = f"{self.model_output_path}/{output_filename}"
            save_model(model, filepath)
            print(f"Content-Based (Title) model saved to {filepath}")
        else:
            print("Error: Game data not loaded. Call load_processed_data() first.")

    def train_content_based_genres_rating(self, output_filename='content_based_genres_rating_model.joblib'):
        """
        Trains the content-based filtering model based on genres and rating score.
        (Corresponds to Recommendation #2 in the notebook)
        """
        if self.df_games is not None:
            print("Training Content-Based (Genres & Rating) model...")
            # Assuming ContentBasedGenresRatingRecommender has a fit method that takes necessary data
            model = ContentBasedGenresRatingRecommender()
            model.fit(self.df_games) # The fit method in the class would handle the TF-IDF and similarity calculations
            filepath = f"{self.model_output_path}/{output_filename}"
            save_model(model, filepath)
            print(f"Content-Based (Genres & Rating) model saved to {filepath}")
        else:
            print("Error: Game data not loaded. Call load_processed_data() first.")

    def train_content_based_annoy(self, output_filename='content_based_annoy_model.annoy'):
        """
        Trains the content-based filtering model based on genres, rating, and Metacritic score using Annoy.
        (Corresponds to Recommendation #3 and #4 in the notebook)
        """
        if self.df_games is not None:
            print("Training Content-Based (Annoy) model...")
            model = ContentBasedAnnoyRecommender() # Assuming this class handles the Annoy index creation in its fit method
            model.fit(self.df_games)
            filepath = f"{self.model_output_path}/{output_filename}"
            model.save(filepath) # Annoy models have their own save method
            print(f"Content-Based (Annoy) model saved to {filepath}")
        else:
            print("Error: Game data not loaded. Call load_processed_data() first.")

    def train_collaborative_filtering_playtime(self, output_filename='collaborative_filtering_playtime_model.joblib'):
        """
        Trains the collaborative filtering model based on total playtime.
        (Corresponds to Recommendation #5 in the notebook)
        """
        if self.df_games_users_all is not None:
            print("Training Collaborative Filtering (Playtime) model...")
            model = CollaborativeFilteringPlaytimeRecommender() # Assuming this class handles user-item matrix creation and similarity calculation
            model.fit(self.df_games_users_all)
            filepath = f"{self.model_output_path}/{output_filename}"
            save_model(model, filepath)
            print(f"Collaborative Filtering (Playtime) model saved to {filepath}")
        else:
            print("Error: User-game interaction data not loaded. Call load_processed_data() first.")

    def train_collaborative_filtering_multi_category(self, output_filename='collaborative_filtering_multi_model.joblib'):
        """
        Trains the collaborative filtering model based on rating, rating score, and total playtime.
        (Corresponds to Recommendation #6 and #7 in the notebook)
        """
        if self.df_games_users_all is not None:
            print("Training Collaborative Filtering (Multi-Category) model...")
            model = CollaborativeFilteringMultiCategoryRecommender() # Assuming this class handles combining and normalizing features
            model.fit(self.df_games_users_all)
            filepath = f"{self.model_output_path}/{output_filename}"
            save_model(model, filepath)
            print(f"Collaborative Filtering (Multi-Category) model saved to {filepath}")
        else:
            print("Error: User-game interaction data not loaded. Call load_processed_data() first.")

    def train_matrix_factorization(self, output_filename='matrix_factorization_model.joblib'):
        """
        Trains the matrix factorization model using the implicit library.
        (Corresponds to Recommendation #10 in the notebook)
        """
        if self.df_users is not None and self.df_games is not None:
            print("Training Matrix Factorization model...")
            model = MatrixFactorizationRecommender() # Assuming this class encapsulates the ALS model training
            model, _, _, _, _ = model.build_model(self.df_users, self.df_games) # Adjust based on actual implementation
            filepath = f"{self.model_output_path}/{output_filename}"
            save_model(model, filepath)
            print(f"Matrix Factorization model saved to {filepath}")
        else:
            print("Error: User and game data not loaded. Call load_processed_data() first.")

    # Add methods for training other recommendation models as needed (e.g., Hybrid, Popularity-Based)

    def run_training(self, models_to_train='all'):
        """
        Orchestrates the training of specified models.

        Args:
            models_to_train (str or list, optional): Specifies which models to train.
                                                    Can be 'all' or a list of model names (e.g., ['content_based_title', 'collaborative_filtering_playtime']).
                                                    Defaults to 'all'.
        """
        self.load_processed_data()
        if self.df_games is not None or self.df_users is not None or self.df_games_users_all is not None:
            if models_to_train == 'all':
                self.train_content_based_title()
                self.train_content_based_genres_rating()
                self.train_content_based_annoy()
                self.train_collaborative_filtering_playtime()
                self.train_collaborative_filtering_multi_category()
                self.train_matrix_factorization()
                # Train other models if implemented
            elif isinstance(models_to_train, list):
                if 'content_based_title' in models_to_train:
                    self.train_content_based_title()
                if 'content_based_genres_rating' in models_to_train:
                    self.train_content_based_genres_rating()
                if 'content_based_annoy' in models_to_train:
                    self.train_content_based_annoy()
                if 'collaborative_filtering_playtime' in models_to_train:
                    self.train_collaborative_filtering_playtime()
                if 'collaborative_filtering_multi' in models_to_train:
                    self.train_collaborative_filtering_multi_category()
                if 'matrix_factorization' in models_to_train:
                    self.train_matrix_factorization()
                # Add conditions for other models
            else:
                print("Invalid value for models_to_train.")
        else:
            print("Error: No data loaded for training.")

if __name__ == "__main__":
    # Example usage:
    trainer = RecommendationTrainer(data_path='processed_data') # Assuming a 'processed_data' directory
    trainer.run_training(models_to_train='all')
    # Or to train specific models:
    # trainer.run_training(models_to_train=['content_based_title', 'matrix_factorization'])