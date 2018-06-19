import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../code'))

from recommend import (
    combine_scores,
    recommend_lang,
    recommend_readme,
    recommend_topic,
    get_feature_weights
)

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for
)

app = Flask(__name__)

@app.route('/404')
def not_found():
    return render_template('404.html')

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/recommend-software', methods=['POST'])
def recommend_software():

    user = request.form['username']

    # Basepath to where the data files are located
    base_path = os.path.abspath(os.path.dirname(__file__) + '/../output')

    user_data = None
    user_file = os.path.abspath(base_path + '/{}.csv'.format(user))

    # If data file for user doesn't exist, redirect to 404 Not found. Otherwise
    # just read the data.
    if os.path.exists(user_file):
        user_data = pd.read_csv(user_file, index_col=0)
    else:
        return redirect(url_for('not_found'), 302)

    repository_file = os.path.abspath(base_path + '/data.csv')

    # Reading the repository-data
    repository_data = pd.read_csv(repository_file, index_col=0, low_memory=False)

    # Reading the preprocessed README-values
    user_readmes = pd.read_csv("{}/{}_tok.csv".format(base_path, user), index_col=0, low_memory=False)
    repository_readmes = pd.read_csv("{}/data_tok.csv".format(base_path), index_col=0, low_memory=False)

    # Calculating similarities per feature
    similarities_lang = recommend_lang(user_data, repository_data)
    similarities_topic = recommend_topic(user_data, repository_data, user)
    similarities_readme = recommend_readme(user_readmes, repository_readmes)

    # Calculate final recommendations
    features = pd.concat((similarities_lang, similarities_topic, similarities_readme), axis=1)
    feature_weights = get_feature_weights(user)
    recommendations = combine_scores(features, feature_weights)
    recommendations = recommendations.head(10)

    return render_template(
        'recommendations.html',
        recommendations=recommendations.head(10),
        total=recommendations.shape[0],
        username=user)
