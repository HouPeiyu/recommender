import pandas as pd
import numpy as np
import helper
import sys
import os

from sklearn.metrics import pairwise_distances


def recommend_lang(user_data, repository_data):
    """
    Make recommendations for user.
    If `user_data` and `repository_data` are not instances of Pandas Series or
    DataFrame, respectively, then raise TypeError.
    Arguments:
    ==========
    user_data: A Pandas series containing users values.
    repository_data: A Pandas DataFrame containing repository information
    Returns:
    ========
    result: A Pandas Series, where indices are repository names and values are
            the scores for the repositories.
    """

    if type(user_data) != pd.Series and type(repository_data) != pd.DataFrame:
        raise TypeError("user_data must be Pandas Series and repository_data must be a Pandas DataFrame")

    # Drop repositories the user already has from repository_data
    repository_data = repository_data.drop(
        np.intersect1d(repository_data.columns, user_data.columns), axis='columns')

    # Reshape user data and repository data
    user_data, repository_data = reshape(user_data, repository_data)

    # Keep only language-related data
    user_data = user_data.loc[user_data.index.str.startswith('l_')]
    repository_data = repository_data.loc[repository_data.index.str.startswith('l_')]

    # Subtract user_data mean from user data to differentiate between the following cases:
    # Bad match: Repo has language A, user does not have language A
    # Neutral match: Neither has language A
    # Without subtraction, cosine similarity gives 0 for both.
    # With subtraction, the bad match gets a lower rating than the neutral match.
    user_data = user_data - user_data.mean()
    lang_similarities = 1 - pairwise_distances(
        repository_data.T,
        user_data.values.reshape(1, -1),
        metric='cosine',
        n_jobs=1)

    lang_similarities = pd.Series(lang_similarities.flatten(), index=repository_data.columns)

    # # Normalize to [0,1]
    # lang_similarities = helper.normalize(lang_similarities)

    return lang_similarities


def recommend_topic(user_data, repository_data, username):
    """
    Make recommendations for user based on repository languages.
    If `user_data` and `repository_data` are not instances of Pandas Series or
    DataFrame, respectively, then raise TypeError.
    Arguments:
    ==========
    user_data: A Pandas series containing users values.
    repository_data: A Pandas DataFrame containing repository information
    Returns:
    ========
    result: A Pandas Series, where indices are repository names and values are
            the scores for the repositories.
    """

    if type(user_data) != pd.Series and type(repository_data) != pd.DataFrame:
        raise TypeError("user_data must be Pandas Series and repository_data must be a Pandas DataFrame")

    # Ensure float data type
    user_data = user_data.drop(['readme']).fillna(0).astype('float')
    repository_data = repository_data.drop(['readme']).fillna(0).astype('float')

    # Drop repositories the user already has from repository_data
    repository_data = repository_data.drop(
        np.intersect1d(repository_data.columns, user_data.columns), axis='columns')

    # Keep only topic-related data
    user_langs, user_topics, repo_langs, repo_topics = helper.get_langs_topics(user_data, repository_data)

    # Calculate Jaccard similarity between user topics and each repository's topics
    topic_similarities = repo_topics.apply(helper.jaccard, user_topics=user_topics, axis=0)

    # # Normalize to [0,1]
    # topic_similarities = helper.normalize(topic_similarities)

    return topic_similarities


def recommend_readme(user_vecs, repository_vecs):
    """
    Make recommendations for user based on README similarity.

    If `user_data` and `repository_data` are not instances of Pandas Series or
    DataFrame, respectively, then raise TypeError.

    Arguments:
    ==========

    user_data: A Pandas series containing users values.
    repository_data: A Pandas DataFrame containing repository information

    Returns:
    ========

    result: A Pandas Series, where indices are repository names and values are
            the scores for the repositories.
    """

    if type(user_vecs) != pd.Series and type(repository_vecs) != pd.DataFrame:
        raise TypeError("user_vecs must be Pandas Series and repository_vecs must be a Pandas DataFrame")

    # Drop repositories the user already has from repository_vecs
    repos_in_common = np.intersect1d(list(repository_vecs.index), list(user_vecs.index))
    repository_vecs = repository_vecs.drop(repos_in_common, axis='index')

    # Compute readme similarities
    readme_similarities = helper.get_readme_sim(user_vecs, repository_vecs)

    # # Normalize to [0,1]
    # readme_similarities = helper.normalize(readme_similarities)

    return readme_similarities


def reshape(user_data, repo_data):
    """
    Collapses user_data to one binary vector containing languages and topics.
    Ensures there's no NA-values in data.
    Converts datatypes to int and binarizes the data
    Reshapes user_data and repo_data to similar dimensions, i.e. both will have same number of features.
    """

    if type(user_data) != pd.DataFrame and type(repo_data) != pd.DataFrame:
        raise TypeError("user_data and repo_data must be Pandas DataFrames.")

    if "readme" in user_data.index:
        user_data = user_data.drop("readme", axis="index")

    if "readme" in repo_data.index:
        repo_data = repo_data.drop("readme", axis="index")

    # Remove NA-values
    user_data = user_data.fillna(0)
    repo_data = repo_data.fillna(0)

    # Collapse user_data to sum of languages and topics and convert to int
    user_data = user_data.sum(axis=1)
    user_data = user_data.astype(float).astype(int)

    # Binarize user_data and repo_data
    user_data[user_data != 0] = 1
    repo_data[repo_data != 0] = 1

    # Rename, so we can access it via column name
    user_data = user_data.rename('User')

    repo_data = pd.concat([repo_data, user_data], axis=1).fillna(0)
    user_data = repo_data.loc[:, 'User'].astype(int)

    repo_data = repo_data.drop('User', axis=1)
    repo_data = repo_data.astype(float).astype(int)

    return user_data, repo_data


def combine_scores(features, feature_weights):

    final_scores = features.dot(feature_weights)
    final_scores = pd.Series(np.array(final_scores.values.reshape(final_scores.size,)))
    final_scores.index = features.index
    final_scores = helper.normalize(final_scores.sort_values(ascending=False))

    return final_scores


# TODO: Get weights from a model built on user feedback
def get_feature_weights(user):

    # Placeholder weights
    weights = pd.Series([0.33, 0.33, 0.33], index=['lang', 'topic', 'readme'])
    weights = weights.values.reshape(weights.size, 1)

    return weights


