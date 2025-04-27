from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    """Renders the input form."""
    return render_template('index.html')

@app.route('/display_game_name', methods=['POST'])
def display_game_name():
    """Handles the form submission and displays the entered game name."""
    game_name = request.form['game_title']
    return render_template('display_game_name.html', game_name=game_name)

if __name__ == '__main__':
    app.run(debug=True)