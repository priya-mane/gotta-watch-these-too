import pandas as pd
import numpy as np
import requests
import json
import gzip
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_concat_data(row):
    return row['genres'] + " " + row['keywords'] + " " + str(row['tagline']) + " " + str(row['overview'])


def get_genre(row):
    s = row['genres']
    # Convert string to list
    lst = s.strip('][').split('}, ')

    # If list is empty return ""
    if lst[0] == "":
        return ""

    # Add "}" to complete dict format
    lst = [k+"}" for k in lst]

    # Remove doublt "}" for the last element
    lst[-1] = lst[-1][:-1]

    # Convert string to dict
    lst = [json.loads(g_string)["name"] for g_string in lst]

    # Join all elements of list to make a string
    g_string = " ".join(lst)
    return g_string


def get_keywords(row):
    s = row['keywords']
    # Convert string to list
    lst = s.strip('][').split('}, ')

    # If list is empty return ""
    if lst[0] == "":
        return ""

    # Add "}" to complete dict format
    lst = [k+"}" for k in lst]

    # Remove doublt "}" for the last element
    lst[-1] = lst[-1][:-1]

    # Convert string to dict
    lst = [json.loads(g_string)["name"] for g_string in lst]

    # Join all elements of list to make a string
    g_string = " ".join(lst)
    return g_string


def make_movies_df():
    # Read both csv files
    df = pd.read_csv("movies.csv")
    credits_df = pd.read_csv("credits.csv")

    # Important features wrt recommender engine
    features = ['original_title', 'genres',
                'keywords', 'tagline', 'overview']

    # Get these features in  new dataframe
    movies_df = df[features]

    # Rename column "original_title" from movies_df to "title" so that
    # it matches with column name of credits_df for the merge operation
    movies_df = movies_df.rename(
        columns={"original_title": "title"}, inplace=False)

    # Merge the daataframes based on the column "title" => to get movies id,cast only from credits.csv
    movies_df = pd.merge(movies_df, credits_df,
                         on='title').drop(columns=['cast'])

    # Extract genre from genre dict object
    movies_df['genres'] = movies_df.apply(lambda x: get_genre(x), axis=1)

    # Extract keywords from keywords dict object
    movies_df['keywords'] = movies_df.apply(lambda x: get_keywords(x), axis=1)

    # Get data req for similarity calculation
    movies_df['whole_data'] = movies_df.apply(
        lambda x: get_concat_data(x), axis=1)

    # Sort data by the name of movie
    movies_df.sort_values(by=['title'], ascending=True, inplace=True)

    # Save the dataframe for further use
    movies_df.to_csv("recommender_features.csv", index=False)


def remove_stop_words(row):
    sentence = row['whole_data']
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_sentence)


def make_similarity_matrix():
    movies_df = pd.read_csv("recommender_features.csv")

    movies_df['whole_data'] = movies_df.apply(
        lambda x: remove_stop_words(x), axis=1)

    cv = CountVectorizer()

    count_matrix = cv.fit_transform(movies_df['whole_data'])
    cosine_sim = cosine_similarity(count_matrix)

    # np.save("cosine_sim", cosine_sim)
    f = gzip.GzipFile("cosine_sim.npy.gz", "w")
    np.save(file=f, arr=cosine_sim)
    f.close()


def get_index_from_title(title):
    df = pd.read_csv("recommender_features.csv")
    return df[df.title == title].index


def get_title_from_index(index):
    df = pd.read_csv("recommender_features.csv")
    return df.loc[index]["title"]


def get_movie_data(movie_title, df):
    movie_details = df[df['title'] == movie_title]
    o = movie_details["overview"].values[0]
    poster = movie_details['poster_url'].values[0]
    movie_obj = {
        "title": movie_title,
        "overview": o,
        "poster": poster
    }
    return movie_obj


def get_top_recommendations(movie_title, top_choices=10):
    # cosine_sim = np.load("cosine_sim.npy")

    f = gzip.GzipFile('cosine_sim.npy.gz', "r")
    cosine_sim = np.load(f)

    movie_index = get_index_from_title(movie_title)
    movies_df = pd.read_csv("recommender_features.csv")

    similar_movies = list(enumerate(cosine_sim[movie_index].flatten()))

    sorted_similar_movies = sorted(
        similar_movies, key=lambda x: x[1], reverse=True)

    recommendations_movie_titles = []

    i = 0
    for element in sorted_similar_movies:
        # print(element)
        recommendations_movie_titles.append(get_title_from_index(element[0]))
        i = i+1
        if i > top_choices:
            break

    recommendations_movie_titles = recommendations_movie_titles[1:]

    recommendations = []

    for title in recommendations_movie_titles:
        recommendations.append(get_movie_data(title, movies_df))

    recommendations_dict = dict()

    i = 0
    for movie in recommendations:
        recommendations_dict[i] = movie
        i += 1

    recommendations_json = json.dumps(recommendations_dict, indent=4)

    return recommendations_json


"""
if __name__ == '__main__':
    # make_movies_df()
    # make_similarity_matrix()
    title = "Spectre"
    df = pd.read_csv("recommender_features.csv")
    lst = get_top_recommendations(title)
    print(lst)
"""
