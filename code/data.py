'''
Github dataset extractor.
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

print('')
print('Github dataset extractor')
print('========================\n')

if len(sys.argv) != 2:
    print('Usage:\n\tpython data.py number-of-repositories\n')
    sys.exit(0)

limit = int(sys.argv[1])

rate_limit() # Inform about rate limits

print('')
print('Repositories are limited to following licenses:\n')
[print(' - {}'.format(x)) for x in ['BSD 2-clause', 'BSD 3-clause', 'MIT']]

print('')
print('Extracting data for {} open source repositories'.format(limit))
print('')

n_retrieved = 0
retrieved_repositories = []

repos, response = resolve_url(
    'https://api.github.com/search/repositories?q=license:bsd-3-clause+license:bsd-2-clause+license:mit')

data = []
repos = repos['items']

while len(repos) > 0:

    data.append(process_repo(repos.pop()))

    n_retrieved += 1
    if n_retrieved % 10 == 0:
        print('{:>4} done'.format(n_retrieved, limit))

    if n_retrieved == limit:
        break

    n = len(repos)
    if n < 1:
        next_url = response.links['next']
        repos, response = resolve_url(next_url['url'])
        repos = repos['items']
        
# Get the directory of current script (code/vectorize.py)
base = os.path.dirname(__file__)

# Create output filename (absolute path)
to_file = os.path.abspath('{}/../output/{}.json'.format(base, 'data'))

# Ensure existence of ../output 
if not os.path.exists('./output/'):
    os.mkdir('./output/')

print('')
print('Saving to {}'.format(to_file))

with open(to_file, 'w') as f:
    json.dump(data, f, indent=2)

print('')
print('All done.')
print('')
