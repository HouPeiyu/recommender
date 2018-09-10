'''
Github user profile extractor.
'''
import json
import sys
import os

from helper import (
    get_languages,
    get_license,
    get_readme,
    get_topics,
    get_user_repos,
    is_contributor,
    process_repo,
    rate_limit,
    resolve_url
)

def get_userdata(user, output=False):

    # Get all user repos: 1) own repos (either fork or non-fork), 2) starred
    # repos, 3) followed repos (subscriptions)
    repos = get_user_repos(user)

    data = []
    skipped = []

    if output == True:
        print('')

    # Extract the data from repos
    for idx, repo in enumerate(repos):

        if output == True:
            print('Processing repo {} of {}'.format(idx + 1, len(repos)))
        entry = process_repo(repo, user)
        if entry['valid'] == True:
            data.append(entry)
        else:
            skipped.append(entry)

    return data, skipped

if __name__ == '__main__':

    print('')
    print('Github user profile extractor')
    print('=============================\n')

    if len(sys.argv) != 2:
        print('Usage:\n\tpython user.py github-user-name\n')
        sys.exit(0)

    user = sys.argv[1]

    rate_limit() # Inform about rate limits

    print('')
    print('Query: {}'.format(user))

    data, _ = get_userdata(user=user, output=True)

    print('')
    print('Found {} repositories related to user {}'.format(len(data), user))
    print('')

    # Get the directory of current script (code/vectorize.py)
    base = os.path.dirname(__file__)

    # Create output filename (absolute path)
    to_file = os.path.abspath('{}/../output/{}.json'.format(base, user))
    print('Saving to `{}`\n'.format(to_file))

    # Ensure existence of ../output
    if not os.path.exists('./output/'):
        os.mkdir('./output/')

    with open(to_file, 'w') as f:
        json.dump(data, f, indent=2)

    print('All done.')
    print('')
