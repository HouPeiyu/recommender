import pandas as pd
import requests as rq
import json
import os
import numpy as np


def process_repo(repo, user = None):

    entry = {}

    # Accepted license for this project
    licenses = ["mit", "bsd-2-clause", "bsd-3-clause"]

    owner, name = repo["owner"]["login"], repo["name"]

    license = get_license(owner, name)

    # Get the specific part of license response, where Github-detected license
    # information is. The actual license text is not needed here, so skipping it
    if "license" in license:
        license = license["license"]

    # The "key" contains license keys specified by Github and we target the ones
    # listed at the beginning of this function. Even if the license is not valid
    # for us, add few bits of info about the specific repo.
    if "key" in license.keys() and license["key"] not in licenses:
        entry["owner"] = owner
        entry["repo"] = name
        entry["valid"] = False
        return entry
    else:
        entry["owner"] = owner
        entry["repo"] = name
        entry["fork"] = repo["fork"]
        entry["license"] = license
        entry["readme"] = get_readme(owner, name)
        entry["topics"] = get_topics(owner, name)
        entry["languages"] = get_languages(owner, name)

        # When retrieving information for open source repositories not related
        # to the user, this information is not needed. But if retrieving info
        # for current user, then this is needed.
        if user != None:
            entry["contributor"] = is_contributor(user, owner, name)

    # If we got this far, the entry is "valid"
    entry["valid"] = True

    return entry


def rate_limit():

    rate_limit, _ = resolve_url("https://api.github.com/rate_limit")

    if rate_limit != None:
        try:
            print("Rate limit: {} of {} remaining".format(
                rate_limit["rate"]["remaining"],
                rate_limit["rate"]["limit"]))
        except: pass
    else:
        print('Failed to resolve URL https://api.github.com/rate_limit')



def resolve_url(url):
    """
    Resolves an URL to JSON-data.

    Uses `GITHUB` environment variable for Github Authorization token.

    Arguments:
    ==========

    url: An URL
    """
    token = os.getenv("GITHUB")

    if token != None:
        response = rq.get(
            url=url,
            headers={
                "Authorization" : "bearer {}".format(token),
                "Accept"        : "application/vnd.github.mercy-preview+json"
            })

        if response.status_code == 404:
            return None, response

        return json.loads(response.text), response
    else:
        print('Environment variable GITHUB not found.\nCreate a personal access token on Github and store it to an environment variable.')
        return None, None


def get_user_repos(user_id):
    """
    Retrieve repositories related to user (user_id). This means following:

    - Users own repositories
    - Repositories user is watching (subscriptions)
    - Repositories user has bookmarked (starred)
    """
    urls = [
        "https://api.github.com/users/{}/repos".format(user_id),
        "https://api.github.com/users/{}/watched".format(user_id),
        "https://api.github.com/users/{}/starred".format(user_id),
        ]

    responses = []
    for url in urls:
        repos, _ = resolve_url(url)
        if repos != None:
            for repo in repos:
                responses.append(repo)

    return responses


def get_languages(owner, repo):
    """
    Get the Github repository languages for specified owner/repo.

    Github endpoint: GET /repos/:owner/:repo/languages

    Returns language - dict pairs; if none, then empty dict. See example below.

    ```
    # Languages:
    {
      "JavaScript": 19877,
      "HTML": 4183,
      "CSS": 3564
    }

    # Empty:
    {

    }
    ```

    Arguments:
    ==========

    owner: Github username
    repo: Repository name

    Return:
    languages: a dict containing language, loc, pairs; if none, then empty dict
    """
    uri = "https://api.github.com/repos/{}/{}/languages".format(owner, repo)
    languages, _ = resolve_url(uri)
    return languages


def get_topics(owner, repo):
    """
    Get the Github topics for specified owner/repo.

    Github endpoint: GET /repos/:owner/:repo/topics

    Arguments:
    ==========

    owner: Github username
    repo: Repository name
    """
    uri = "https://api.github.com/repos/{}/{}/topics".format(owner, repo)
    topics, _ = resolve_url(uri)

    # Current Github API returns a dict as follows:
    # {"names": ["topic1", "topic2", "..."]}
    if topics != None and "names" in topics:
        topics = topics["names"]
    else:
        topics = []

    return topics


def get_license(owner, repo):
    """
    Get the license for specified owner/repo.

    Github endpoint: GET /repos/:owner/:repo/license

    Arguments:
    ==========

    owner: Github username
    repo: Repository name
    """
    uri = "https://api.github.com/repos/{}/{}/license".format(owner, repo)
    license, _ = resolve_url(uri)

    if license == None:
        return {}

    return {
        "encoding": license["encoding"],
        "content": license["content"],
        "license": license["license"]}


def get_readme(owner, repo):
    """
    Get the README for specified owner/repo.

    Github endpoint: GET /repos/:owner/:repo/contents/{+path}

    Arguments:
    ==========

    owner: Github username
    repo: Repository name
    """
    uri = "https://api.github.com/repos/{}/{}/readme".format(owner, repo)
    readme, _ = resolve_url(uri)

    if readme == None:
        return {}

    return { "encoding": readme["encoding"], "content": readme["content"] }


def is_contributor(user, owner, repo):
    uri = "https://api.github.com/repos/{}/{}/contributors".format(owner, repo)
    contributors, _ = resolve_url(uri)
    if contributors == None:
        return False

    for contributor in contributors:
        if contributor["login"] == user:
            return True
    return False


def normalize(df):
    # Normalizes to [0,1]
    # Returns every value - min divided by the range, i.e.
    # df_i - df.min() / df.range()
    # Note: df is in practice always a Pandas Series
    rng = df.max().max() - df.min().min()
    if rng == 0:
        # raise ValueError("helper.normalize: range 0, division by zero")
        return df
    df = df - df.min().min()
    return df / float(rng)


def normalize_combo_score(df):
    minimum = 0
    rng = df.max().max() - minimum
    if rng == 0:
        # raise ValueError("helper.normalize: range 0, division by zero")
        return df
    df = df - minimum
    return df / float(rng)


def l2_norm(vec):
    if vec.size == vec.shape[0]:
        vec = vec.values.reshape((vec.size,))
    result = np.sqrt(vec.dot(vec.T))
    if result.shape == (1,1):
        return result[0][0]
    else:
        return result


def jaccard(repo_topics, user_topics):
    intersect = user_topics[repo_topics != 0].sum()
    union = user_topics.sum() + repo_topics.sum() - intersect
    # If no topics marked for either, union and intersect will both be zero:
    if union - intersect == 0:
        return 0
    else:
        return intersect / float(union)


def get_readme_sim(user_vecs, repository_vecs):
    # Compute an average user README vector to compare against other's repository readmes
    user_mean_vec = normalize(user_vecs.mean(axis=0))
    user_mean_vec = user_mean_vec.values.reshape((1, user_mean_vec.size))

    dot_products = repository_vecs.dot(user_mean_vec.T)
    dot_products = dot_products.values.reshape((dot_products.size))
    lengths = (l2_norm(user_mean_vec) * repository_vecs.apply(l2_norm, axis=1))

    return dot_products / lengths


def get_langs_topics(user_data, repository_data):
    """ Create dataframes for user/repo languages and topics """
    username = username = user_data.columns[0].split('/')[0]

    # Count how many repositories user has in each of the languages and topics:
    counts = user_data.agg(func="sum", axis=1)
    user_langs = pd.DataFrame(counts[counts.index.str.startswith("l_")], columns=[username])
    user_topics = pd.DataFrame(counts[counts.index.str.startswith("t_")], columns=[username])

    # Get repo languages and topics
    repo_langs = repository_data[repository_data.index.str.startswith('l_')]
    repo_topics = repository_data[repository_data.index.str.startswith('t_')]

    # Binarize user topics
    user_topics[user_topics > 0] = 1

    # Make user and repository vector lengths equal
    repo_langs = pd.concat([user_langs, repo_langs], axis=1).fillna(0)
    user_langs = repo_langs[username]
    repo_langs = repo_langs.drop([username], axis='columns')
    repo_topics = pd.concat([user_topics, repo_topics], axis=1).fillna(0)
    user_topics = repo_topics[username]
    repo_topics = repo_topics.drop([username], axis='columns')

    return (user_langs, user_topics, repo_langs, repo_topics)

