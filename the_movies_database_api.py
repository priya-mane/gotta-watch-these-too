import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import os


def get_poster_paths(movie_id):
    key = os.environ.get('API_KEY')
    query = "https://api.themoviedb.org/3/movie/" + \
        str(movie_id) + "?api_key=" + key
    movie = requests.get(query)

    image_url = ""

    try:
        if movie is not None:
            file_path = movie.json()["poster_path"]
            if file_path is not None:
                image_url = "https: // image.tmdb.org/t/p/w500" + file_path

    except:
        image_url = ""

    return image_url


if __name__ == '__main__':
    df = pd.read_csv("recommender_features.csv")

    tqdm.pandas(desc='poster_url')
    df['poster_url'] = df['movie_id'].progress_apply(
        lambda x: get_poster_paths(x))

    df.to_csv("recommender_features.csv", index=False)

    # url for no image : https://ibb.co/zmTCCqg
