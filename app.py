from flask import Flask, redirect, url_for, render_template
import pandas as pd
from flask_wtf.csrf import CSRFProtect
from flask import request
import recommendation_engine
import json


app = Flask(__name__)
csrf = CSRFProtect(app)
csrf.init_app(app)

app.config['SECRET_KEY'] = "priyamane"


@app.route('/')
def index():
    movies_list = pd.read_csv("recommender_features.csv")['title'].tolist()
    return render_template('index.html', movies=movies_list)


@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    req = request.json
    user_movie = req.get('user_movie')

    recommendations = recommendation_engine.get_top_recommendations(user_movie)

    return recommendations


if __name__ == '__main__':
    app.run(debug=True)
