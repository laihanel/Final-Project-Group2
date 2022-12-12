import numpy as np
import pandas as pd
import re
import nltk
import spacy
import string
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.casual import casual_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet



# sub1 = pd.read_csv("MacOS_submission.csv")
# sub2 = pd.read_csv("windows_submission.csv")
sub1 = pd.read_csv(r'../Data/MacOS_submission.csv', encoding='utf-8')
sub2 = pd.read_csv(r'../Data/windows_submission.csv', encoding='utf-8')

# com1 = pd.read_csv("MacOS_comments.csv")
# com2 = pd.read_csv("windows_comments.csv")
com1 = pd.read_csv(r'../Data/windows_comments.csv', encoding='utf-8')
com2 = pd.read_csv(r'../Data/MacOS_comments.csv', encoding='utf-8')

# Removes everything but the features we are interested in.
sub1 = sub1[['author', 'created_utc', 'id', 'num_comments', 'over_18',
             'score', 'subreddit', 'title', 'selftext']]
sub2 = sub2[['author', 'created_utc', 'id', 'num_comments', 'over_18',
             'score', 'subreddit', 'title', 'selftext']]
com1 = com1[['author', 'created_utc',
             'parent_id', 'score', 'subreddit', 'body']]
com2 = com2[['author', 'created_utc',
             'parent_id', 'score', 'subreddit', 'body']]

submission = pd.concat([sub1, sub2],ignore_index=True)
# submission.columns = submission.columns.str.replace('selftext', 'body')
submission['body'] = submission['title'] + submission['selftext']
comments = pd.concat([com1, com2], ignore_index=True)
# =================================================================================

## drop removed and empty post
submission = submission.drop(submission[(submission['body'] == '[removed]') | (submission['body'] == '')].index)
submission = submission.dropna()
print(submission.head())

comments = comments.drop(comments[(comments['body'] == '[removed]') | (comments['body'] == ' ')].index)
comments = comments.dropna()
print(comments.head())

def decontracted(df):
    df['body'] = df['body'].str.replace(r'\'m', ' am')
    df['body'] = df['body'].str.replace(r"won't", "will not")
    df['body'] = df['body'].str.replace(r"can\'t", "can not")
    df['body'] = df['body'].str.replace(r"n\'t", " not")
    df['body'] = df['body'].str.replace(r"\'re", " are")
    df['body'] = df['body'].str.replace(r"\'d", " would")
    df['body'] = df['body'].str.replace(r"\'ll", " will")
    df['body'] = df['body'].str.replace(r"\'ve", " have")
    df['body'] = df['body'].str.replace(r"\'s", " is")
    return df

submission = decontracted(submission)
comments = decontracted(comments)

def tokenization(text):
    """custom function to tokenization the list"""
    # tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
    tokenizer = TreebankWordTokenizer()
    return tokenizer.tokenize(text)
    # return casual_tokenize(text)

submission["tokenized"] = submission['body'].apply(lambda text: tokenization(text))
comments["tokenized"] = comments['body'].apply(lambda text: tokenization(text))
# submission.head()

print('-'*20, 'Tokenization Finished', '-'*20)

# PUNCT_TO_REMOVE = string.punctuation

PUNCT_TO_REMOVE = '!#$%&()*+,-/:<=>@[\\]^_`{|}~'
STOPWORDS = set(stopwords.words('english'))

# # STOPWORDS = set(sklearn_stop_words)
def removePunctStop(text):
    words = []
    for word in text:
        # change to lower case
        word_lower = word.lower()
        # remove punctuations
        if word_lower not in PUNCT_TO_REMOVE and word_lower not in STOPWORDS:
        # if word_lower not in STOPWORDS:
            words.append(word_lower)
        else:
            continue
    return words

submission["text_wo_stop"] = submission['tokenized'].apply(lambda text: removePunctStop(text))
comments["text_wo_stop"] = comments['tokenized'].apply(lambda text: removePunctStop(text))

print('-'*20, 'Stopwords, Punctuations removed', '-'*20)

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text)
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

submission.drop(["body", "tokenized", 'title', 'selftext'], axis=1, inplace=True)
comments.drop(["body", "tokenized"], axis=1, inplace=True)

submission["body"] = submission["text_wo_stop"].apply(lambda text: lemmatize_words(text))
submission.drop(["text_wo_stop"], axis=1, inplace=True)

comments["body"] = comments["text_wo_stop"].apply(lambda text: lemmatize_words(text))
comments.drop(["text_wo_stop"], axis=1, inplace=True)
print('-'*20, 'Lemmatization Finished', '-'*20)

submission = submission.dropna()
comments = comments.dropna()

print(submission.head())
print("Submission data length:" + str(len(submission)))

print(comments.head())
print("Comments data length:" + str(len(comments)))

print('-'*20, 'Saving Results', '-'*20)
submission.to_csv("../Data/submissions_tokenized.csv")
comments.to_csv("../Data/comments_tokenized.csv")