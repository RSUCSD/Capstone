# Video Game Recommender System Project

This document outlines the steps and methodologies used to build a basic video game recommender system, leveraging the Steam dataset from Kaggle. This project explores various approaches to provide personalized game suggestions.

## 1. Project Goal

The primary goal of this project is to build a functional recommender system for video games. The system utilizes machine learning concepts, such as **cosine similarity**, to identify games that are similar to those a user has previously enjoyed [1]. It's important to note that this system initially focuses on a simplified approach, and more advanced techniques could be explored for enhanced accuracy and sophistication [1].

## 2. Data Sources

This project utilizes **Steam datasets from Kaggle** [2]. These datasets contain information on games, users, recommendations, and ratings [2]. Specific datasets explored include:

*   Size M: [https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam/data](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam/data) [2-4]
*   Size L: [https://www.kaggle.com/datasets/whigmalwhim/steam-releases/data](https://www.kaggle.com/datasets/whigmalwhim/steam-releases/data) [2-4]
*   Size XL (1): [https://www.kaggle.com/datasets/maestrocor/steam-games-and-user-playtime-data/data](https://www.kaggle.com/datasets/maestrocor/steam-games-and-user-playtime-data/data) [2-4]
*   Size XL (2): [https://www.kaggle.com/datasets/praffulsingh009/steam-video-games-2024](https://www.kaggle.com/datasets/praffulsingh009/steam-video-games-2024) [2-4]

While other datasets were initially considered, the project focused on the ones listed above [5].

## 3. Libraries Used

The following essential Python libraries were imported and utilized for data manipulation, analysis, and machine learning [6, 7]:

*   **Pandas**: For efficient data handling and manipulation [6, 7].
*   **NumPy**: For numerical computations [6, 7].
*   **mlxtend**: Provides tools for frequent pattern mining [6, 7].
*   **sklearn (scikit-learn)**: Offers tools for text vectorization (**CountVectorizer**, **TfidfVectorizer**), similarity calculations (**cosine_similarity**, **linear_kernel**), linear models (**LinearRegression**), model selection (**train_test_split**), nearest neighbors (**NearestNeighbors**), and preprocessing (**LabelEncoder**) [6, 7].
*   **scipy**: Provides tools for sparse matrix operations (**csr_matrix**, **coo_matrix**) [6, 7].
*   **kagglehub**: Facilitates dataset retrieval from Kaggle directly [3, 6, 7].
*   **annoy**: For approximate nearest neighbor search [7-9].
*   **dask**: For parallel computing with dataframes [7].
*   **implicit**: For collaborative filtering using matrix factorization (**AlternatingLeastSquares**) [7, 10].
*   **matplotlib**: For data visualization [7, 11-14].
*   **os**: For interacting with the operating system (e.g., listing files) [3, 7].
*   **gc**: For garbage collection (memory management) [7].

## 4. Data Loading and Preprocessing

The project involved several steps to load and preprocess the data [15, 16]:

*   **Downloading Datasets**: Kaggle datasets were downloaded using `kagglehub.dataset_download()` [3].
*   **Memory Management**: Functions like `reduce_memory` were used to optimize memory usage by changing data types [4, 15]. A `data_generator` function was also introduced for loading data in chunks [4, 15].
*   **Loading DataFrames**: Selected CSV files were read into Pandas DataFrames (`df_games_1`, `df_games_2`, `df_games_3`, `df_users`, `df_recommendations`) [4].
*   **Data Analysis**: Preliminary data exploration was performed using `.shape`, `.dtypes`, `.info()`, and `.columns` to understand the structure and characteristics of the datasets [17, 18]. The `.head()` function was used to get a quick overview of the data [17].
*   **Feature Engineering**:
    *   Irrelevant columns were identified and removed from the DataFrames [19, 20].
    *   Columns were renamed for consistency (e.g., 'game' to 'title', 'link' to 'app_id') [21].
    *   Columns were reordered for better organization [22].
    *   Data types were converted as needed (e.g., 'app_id' to `int32`) [22].
    *   Multiple game datasets (`df_games_1`, `df_games_2`, `df_games_3`) were merged based on common columns like 'app_id' and 'title' using `pd.merge()` [23].
    *   Invalid and missing values (NaN) were checked using `.isnull().values.any()`, `.isna().any()`, `.isnull().sum()`, and the columns with missing values were identified [24].
    *   Users with zero total playtime were removed from the `df_users` DataFrame [25].
    *   The `df_games` and `df_users` DataFrames were merged into `df_games_users_all` [25].
*   **Data Visualization**: Top games were visualized based on:
    *   Highest Metacritic Score (bar plots) [11].
    *   Highest Number of Ratings Received (bar plots) [12].
    *   Highest Total Playing Time (bar plots) [13, 14].

## 5. Recommendation Models Implemented

Several recommendation model approaches were explored:

### 5.1. Content-Based Filtering

*   **Based on Game Title**: Used **TF-IDF** to vectorize game titles and **cosine similarity** to find similar games [14, 26-28].
*   **Based on Genres and Rating Score**: Used **TF-IDF** for genres and combined it with normalized rating scores using weighted **cosine similarity** [28-32].
*   **Based on Genres, Rating Score, and Metacritic Score (with Annoy)**: Used **TF-IDF** for genres and combined it with normalized rating and Metacritic scores. An **Annoy index** was built for efficient similarity search [8, 33-37].
*   **Based on Genres, Rating, Rating Score, and Metacritic Score (with Annoy)**: Similar to the above, but also incorporated the 'rating' column by using **Label Encoding** and normalization before combining features and building the **Annoy index** [9, 37-42].

### 5.2. Collaborative Filtering

*   **Based on Total Playtime**: Created a user-item matrix based on 'total_playtime' and used **cosine similarity** to find similar users, then recommended games played by those users [42-45].
*   **Based on Rating, Rating Score, and Total Playtime**: Mapped string ratings to numerical scores, normalized 'total_playtime' and 'rating_score', created a combined score, built a user-item matrix, used **cosine similarity** for user-user similarity, and generated recommendations from similar users [46-51].
*   **Based on Rating, Rating Score, and Top 5 Highest Playing Time**: Similar to the above, but first filtered the data to keep only the top 5 highest played games for each user before building the collaborative filtering model [52-57].

### 5.3. Hybrid Model

*   **Based on Genres, Rating, Rating Score, and Top 10 Highest Playing Time**: Combined content-based (genres and ratings) and collaborative filtering approaches. Used **TF-IDF** for genres, normalized scores, and built a user-item matrix for **cosine similarity** based recommendations from similar users who played the top 10 games [57-64].

### 5.4. Popularity-Based Recommendation

*   Suggested games based on overall engagement metrics like total playtime, average playtime, and the number of unique players. A weighted popularity score was calculated to rank games [65-69].

### 5.5. Matrix Factorization Model

*   Implemented a recommendation system using the **Alternating Least Squares (ALS)** algorithm from the `implicit` library. A user-item matrix (total playtime) was factorized to learn user and game latent factors for generating personalized recommendations [10, 69-72].

## 6. Deployment Structure (Conceptual)

*(Based on the "Deployment Structure Model.pdf" source, this outlines a potential structure for deploying such a system, though not the direct work we've done in the notebook so far):*

A potential deployment structure involves several key components [73-75]:

*   **User/App**: Represents the user interface through which users interact with the recommender system by sending requests (e.g., browsing a website, using a mobile app) [73, 74].
*   **main.py (Entry Point/API)**: Serves as the main entry point of the system. It receives user requests, loads data, orchestrates the model training or loading process, uses trained models to generate recommendations, and returns these recommendations to the user/app. It might use frameworks like Flask or FastAPI for defining API endpoints [73, 74].
*   **data\_preprocessing.py (Data Handling)**: Focuses on loading and preprocessing the data, including cleaning, handling missing values, and feature engineering. It is called by `main.py` [73, 74].
*   **training\_orchestrator.py (Model Management)**: Manages the training process of the recommendation models, potentially including logic for model selection, hyperparameter tuning, and saving/loading trained models. It imports and uses the model implementations from `recommendation_models.py` [73, 75].
*   **recommendation\_models.py (Recommendation Algorithms)**: Contains the implementations of different recommendation algorithms (e.g., content-based, collaborative filtering). These models include methods for training (`fit`) and generating recommendations (`get_recommendations`) [73, 75].
*   **Trained Models**: Represents the saved output of the model training process (e.g., pickle files, joblib files for trained models like Annoy indices or ALS models). These are loaded by `main.py` or `training_orchestrator.py` to generate recommendations [73, 75].
