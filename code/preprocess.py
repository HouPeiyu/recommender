import re
import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk                                
nltk.download('averaged_perceptron_tagger')

def tokenize(line, minlength):
    """
    Remove special characters, links, words below minlength and 
    content enclosed in square, angle and curly brackets from string.

    Arguments:
    ==========

    line: README string
    minlength: minimum number of characters needed to preserve a word
    """
    if type(line) == str:
        line = line.replace('\n', '').replace('\r', '').replace('"', '\'').replace('\'', '').replace('`','')
        line = line.lower()

        # # Remove patterns (more may be added)
        # p_url = re.compile('^.?http')
        # p_git = re.compile('^.?git')
        # p_at = re.compile('.*@.*')
        # p_exc = re.compile('^.?[!;]')
        # p_bslash = re.compile('.*[\\\].*')
        # p_fslash = re.compile('.*[/:|$*#()].*')

        patterns = re.compile('^.?http|''^.?git|''.*@.*|''^.?[!;]|''.*[\\\].*|''.*[/].*|''.*[/:|$*#()].*')

        # Remove bracketed parts:
        line = remove_bracketed(line, brackets=[('[', ']'), ('<','>'), ('{','}'), ('(http', ')')])

        pattern_free = []
        pattern_free.append([word for word in line.split() if not patterns.match(word)])
        line = ' '.join(pattern_free[0])

        # Remove the rest of the special characters
        keepchars = u' 0123456789abcdefghijklmnopqrstuvwxyz'
        keeper = []
        keeper.append([ch if (ord(ch) < 128) else ' ' for ch in line])
        line = ''.join(keeper[0])
        keeper = []
        keeper.append([ch for ch in line if ch in keepchars])
        keeper = ' '.join(''.join(keeper[0]).split()).strip()

        # Remove words shorter than minlength
        keeper = ' '.join([x for x in keeper.split() if len(x) >= minlength])
        return keeper
    else: return ''

def remove_bracketed(line, brackets):
    """
    Remove content between listed brackets. 

    Arguments:
    ==========

    line: README string
    brackets: tuples of start/end characters/strings.
    """

    for bracket in brackets:
        while line.find(bracket[0]) != -1 and line.find(bracket[1]) != -1:
            # Remove inverted brackets
            if line.find(bracket[0]) > line.find(bracket[1]):
                line = line.replace(bracket[0], ' ')
                line = line.replace(bracket[1], ' ')
            # Remove content between valid brackets
            else: line = line[:line.index(bracket[0])] + line[line.index(bracket[1])+1:]
    return line

def filter_pos(text, pos):
    """Returns only selected parts of speech. Parameter pos: string of POS tags"""
    pos_tagged = nltk.pos_tag(text.split())
    return ' '.join([word[0] for word in pos_tagged if word[1] in pos.split()])

def tfidf(user_readmes, repository_readmes, nb_features):
    """Returns vector representations of user's READMEs and repository READMEs"""
    all_readmes = pd.concat([user_readmes, repository_readmes], axis=0)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=nb_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(all_readmes)
    feature_names = tfidf_vectorizer.get_feature_names()

    tfidf_dense = tfidf.todense()
    user_docvecs = pd.DataFrame(tfidf_dense[:user_readmes.shape[0]], index=user_readmes.index, columns=feature_names)
    repository_docvecs = pd.DataFrame(tfidf_dense[user_readmes.shape[0]:], index=repository_readmes.index, columns=feature_names)

    return user_docvecs, repository_docvecs


if __name__ == '__main__':    
    print('')
    print('Github data README preprocessor')
    print('======================\n')

    if len(sys.argv) != 3:
        print('Usage:\n\tpython preprocess.py user.csv data.csv\n')
        sys.exit(0)

    in_1 = sys.argv[1]
    in_2 = sys.argv[2]
    
    print('Reading `{}` and `{}` as input files\n'.format(in_1, in_2))

    user_readmes = pd.read_csv(in_1, index_col=0, low_memory=False).fillna(0).loc['readme']
    repository_readmes = pd.read_csv(in_2, index_col=0, low_memory=False).fillna(0).loc['readme']
    
    # Tokenize readmes
    user_readmes = user_readmes.apply(tokenize, minlength=3)
    repository_readmes = repository_readmes.apply(tokenize, minlength=3)

    # Part-of-speech (POS) tagging: 
    # Preserve only selected parts of speech
    # verbs = u'VB VBG'
    # nouns = u'NN NNS NNP NNPS' 
    nouns = u'NN' 
    print('POS tagging words from {}, please wait...'.format(in_1))
    user_readmes = user_readmes.apply(filter_pos, pos=nouns)
    print('POS tagging words from {}, please wait...'.format(in_2))
    repository_readmes = repository_readmes.apply(filter_pos, pos=nouns)

    # Apply TF-IDF transform to mitigate the effect of words that appear in most readmes
    user_docvecs, repository_docvecs = tfidf(user_readmes, repository_readmes, nb_features=3000)

    out_1 = './output/{}'.format(in_1.split('/')[-1].split('.')[-2]) + '_tok.csv'
    out_2 = './output/{}'.format(in_2.split('/')[-1].split('.')[-2]) + '_tok.csv'

    print('Saving data to `{}` and `{}`\n'.format(out_1, out_2))
    print('Saved data has shapes of {} and {}\n'.format(user_docvecs.shape, repository_docvecs.shape))

    user_docvecs.to_csv(out_1)
    repository_docvecs.to_csv(out_2)

    print('All done.')
    print('')

