# poeml_utility.py
# Description:  functionality for poeml project


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import spacy

# connect to data base
def connect_db():
    from sqlalchemy import create_engine
    dbname = 'poetry_db'
    username = 'ctoews'
    engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
    return engine

# read raw text file to get lines
def parse_sonnets_by_line(pathname, filename):
    # default path:  ../Data/Shakespeare/sonnets.txt
    columns = ['title','line']
    sonnet_lines = []
    f = open("pathname"+"filename","r")
    title = ""
    snippet = ""
    for line in f:
        if len(line)==1:
            continue
        if (len(line)>1)&(len(line)<15):
            title = line.strip('\n\t \ufeff')
            continue
        else:
            snippet = line.strip('\n\t ')
            sonnet_lines.append([title,snippet])

    sonnet_lines = pd.DataFrame(sonnet_lines,columns=['title','line'])

    return sonnet_lines


# read in clean csv file to get full poems
def parse_sonnets_by_poem():
    import pandas as pd
    sonnet_full = pd.read_csv('../Data/Shakespeare/sonnets.csv',names=["title","poem"])

    sonnet_full['title'] = sonnet_full['title'].str.strip('\n\t \ufeff')
    sonnet_full['poem'] = sonnet_full['poem'].str.strip('\n\t ')

    return sonnet_full


# rip out sentneces from a df with full poems
def parse_sonnets_by_sentence(sonnet_full):
    sonnet_sentence = pd.DataFrame(sonnet_full.poem.str.strip().str.split('[.?:]').tolist(), index=sonnet_full.title).stack()
    sonnet_sentence=pd.DataFrame(sonnet_sentence)
    sonnet_sentence.reset_index(inplace=True)
    sonnet_sentence.columns=['title','level','sentence']
    sonnet_sentence['poem']=sonnet_sentence.sentence.str.strip()
    sonnet_sentence = sonnet_sentence.loc[sonnet_sentence.sentence != '',:]
    sonnet_sentence.reset_index(inplace=True)
    sonnet_sentence = sonnet_sentence[['title','sentence']]

    return parse_sonnets_by_sentence

# Every step in a pipeline needs to be a "transformer".
# Define a custom transformer to clean text using spaCy
class CleanTextTransformer(TransformerMixin):
    """
    Convert text to cleaned text
    """

    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# A custom function to clean the text before sending it into the vectorizer
def cleanText(text):

    # import a dictionary of English contractions from another file
    from contractions import contractions_dict

    # replace the contractions with their expanded form
    for contraction, expansion in contractions_dict.items():
        text = text.replace(contraction.lower(),expansion.lower())

    # get rid of newlines
    text = text.strip().replace("\n", " ").replace("\r", " ").replace("-"," ")

    # lowercase
    text = text.lower()

    return text

# A custom function to tokenize the text using spaCy
# and convert to lemmas
def tokenizeText(sample,parser=spacy.load('en')):

    # get the tokens using spaCy
    tokens = parser(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip()
                      if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")

    return tokens
