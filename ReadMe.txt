README.TXT

Project Title: Steam Video Game Recommendation System

Description:
This project is a Flask-based web application that provides video game recommendations based on a user-entered game title. The system uses a combined recommendation approach, considering both the similarity of game titles and their rating scores to generate relevant suggestions.

Files Included:
-   index.html: The main HTML file for the web application, containing the input form for game titles and displaying initial game suggestions.
-   recommendations.html: HTML file to display the list of recommended games.
-   app.py: The Flask application Python script that handles user input, generates recommendations using a pre-trained model, and renders the appropriate HTML pages.
-   Requirements.txt: A list of Python packages required to run the application.
-   /data/df_games.csv: CSV file containing the game data (Note: Path specified in `app.py`).
-   /models/tfidf_vectorizer_rec2.pkl: Pickle file containing the pre-trained TF-IDF vectorizer model (Note: Path specified in `app.py`).
-   /static/: Directory containing static files such as images (e.g., game cover images).

Requirements:
-   Python 3.x
-   Flask
-   pandas
-   scikit-learn
-   Other packages listed in `Requirements.txt`

Installation:
1.  Ensure Python 3.x is installed.
2.  Install the required Python packages using pip:
    ```bash
    pip install -r Requirements.txt
    ```
3.  Place the data file (`df_games.csv`) and the model file (`tfidf_vectorizer_rec2.pkl`) in the locations specified in `app.py` or update the file paths in the script.
4.  The static files (images) should be placed in the `/static/` directory.

Usage:
1.  Run the `app.py` script to start the Flask application:
    ```bash
    python app.py
    ```
2.  Open a web browser and navigate to the specified address (usually http://127.0.0.1:5000/).
3.  Enter a game title in the input form on the `index.html` page and submit.
4.  The application will display a list of recommended games on the `recommendations.html` page.

Limitations:
1.  **Genre Complexity:** The dataset includes numerous variables under the "genres" category. This high dimensionality and complexity in genre representation can make it challenging to accurately compare games and may affect the precision of recommendations. The model might struggle to effectively differentiate between games with highly specific or overlapping genre classifications.
2.  **Missing Data:** The dataset has missing values, particularly for Metacritic scores. The absence of complete rating information can limit the robustness of the recommendation system, as the model relies on these scores to assess game quality and similarity. Recommendations for games with missing Metacritic scores may be less reliable.
3.  **Computational Constraints:** The model training process was constrained by the computational resources available, specifically the RAM limitations of Google Colab. This limitation can restrict the complexity of the model and the size of the dataset that can be effectively processed. Consequently, it may be difficult to develop a more robust machine learning model that incorporates a wider range of game features or more sophisticated algorithms. The model's performance and accuracy could potentially be improved with access to greater computational resources.
4.  **Data Dependency:** The quality and accuracy of the recommendations are highly dependent on the quality and completeness of the underlying game data. Any biases or inaccuracies present in the data will likely be reflected in the model's output.
5.  **Cold Start Problem:** The system may face challenges in providing accurate recommendations for new or less popular games with limited data. This "cold start problem" is a common limitation in recommendation systems.

Notes:
-   Game title input may be case-sensitive.
-   The recommendation system uses a combination of title similarity and rating scores.
-   The "full list of games" link in `index.html` points to the project's data directory on GitHub. Make sure the path is correct if you change the data source.