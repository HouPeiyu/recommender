'''
Vectorizing data retreved from Github.
'''

import sys
import os
import json
import pandas as pd
import numpy as np
import base64

def vectorize(data, output_file=None):

    if type(data) != pd.DataFrame:
        raise TypeError('Input for `vectorize` must be a Pandas DataFrame.')

    data = data[['owner', 'repo', 'languages', 'topics', 'readme']]
    
    df = pd.DataFrame()
    for idx, row in data.iterrows():

        # Topics
        topics = ['t_' + topic.lower() for topic in row[3]]
        topics = pd.DataFrame(np.ones(len(topics)), index=topics)

        # Languages
        languages = row[2]

        # Handle the case when there's no known languages informed
        if not languages:
            languages = {'Unknown': 0}

        # Create languages dataframe
        languages = pd.DataFrame.from_dict(languages, orient='index')
        languages.index = 'l_' + languages.index.str.lower()

        # READMEs
        if 'encoding' in row['readme'] and row['readme']['encoding'] == 'base64':
            readme = base64.b64decode(row['readme']['content'])
        else:
            readme = None
        readme = pd.DataFrame(data=[readme], index=['readme'])

        # Create one entry
        entry = pd.concat([languages, topics, readme])
        entry.columns = [row[0] + '/' + row[1]]
        # entry = entry.iloc[:, 0] # INKA REMOVED

        df = pd.concat([df, entry], axis=1)

    return df

if __name__ == '__main__':
    
    print('')
    print('Github data vectorizer')
    print('======================\n')

    if len(sys.argv) != 2:
        print('Usage:\n\tpython vectorize.py input.json\n')
        sys.exit(0)

    input_file = sys.argv[1]
    print('Reading `{}` as input file\n'.format(input_file))
    data = pd.read_json(input_file)

    df = vectorize(data)

    # Get the directory of current script (code/vectorize.py)
    base = os.path.dirname(__file__)

    # If called like this: python code/vectorize.py input.json, takes only the input
    # filename, input.json at this example.
    input_file = os.path.basename(input_file)

    # Create output filename (absolute path)
    to_file = os.path.abspath('{}/../output/{}.csv'.format(base, input_file.split('.')[0]))

    print('Saving data to `{}`\n'.format(to_file))
    print('Saved data has shape of {}\n'.format(df.shape))
    with open(to_file, 'w') as f:
        df.to_csv(f)
    print('All done.')
    print('')
