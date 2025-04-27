<!DOCTYPE html>
<html>
<head>
    <title>RS Capstone Project - Video Game Recommendation</title>
    <style>
        body {
            background-color: black;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
        }
        h1 {
            margin-top: 80px; /* Increased margin-top */
            margin-bottom: 0;
        }
        h2 {
            margin-top: 20px; /* Increased margin-top */
            font-size: 1.5em;
            font-weight: normal;
        }
        form {
            margin-top: 60px; /* Increased margin-top */
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        input[type="text"] {
            padding: 12px 20px;
            width: 300px;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-bottom: 10px;
            background-color: #222;
            color: #eee;
        }
        input[type="submit"] {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 15px;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
        #game_suggestions {
            margin-top: 30px; /* Increased margin-top */
            font-style: italic;
            color: #ddd;
            margin-bottom: 20px;
        }
        .game_images {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
            max-width: 560px;
            margin-left: auto;
            margin-right: auto;
        }
        .game_images img {
            width: 150px;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        }
    </style>
</head>
<body>
    <h1>RS Capstone Project</h1>
    <h2>Steam Video Game Recommendation</h2>
    <p id="game_suggestions">Here are some famous video games:</p>
    <div class="game_images">
        <img src="{{ url_for('static', filename='batman.jpg') }}" alt="Arkham Knight">
        <img src="{{ url_for('static', filename='gta.jpg') }}" alt="Grand Theft Auto">
        <img src="{{ url_for('static', filename='ac.jpg') }}" alt="Assassin's Creed">
        <img src="{{ url_for('static', filename='fortnite.jpg') }}" alt="Fortnite">
        <img src="{{ url_for('static', filename='twd.jpg') }}" alt="The Walking Dead">
        <img src="{{ url_for('static', filename='cod.jpg') }}" alt="Call of Duty">
        <img src="{{ url_for('static', filename='minecraft.jpg') }}" alt="Minecraft">
        <img src="{{ url_for('static', filename='re.jpg') }}" alt="Resident Evil">
        <img src="{{ url_for('static', filename='mix.jpg') }}" alt="Various">
    </div>
    <form method="POST" action="/display_game_name">
        <input type="text" name="game_title" placeholder="Enter game name">
        <input type="submit" value="Display">
    </form>
</body>
</html>
