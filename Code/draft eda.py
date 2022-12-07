# =================================================================================
import pandas as pd
import torch
from torchtext.legacy import data
import nltk
from nltk import word_tokenize, sent_tokenize
from torchtext.legacy import datasets
import torch.nn as nn
import torch.optim as optim
import time
import seaborn as sns
import matplotlib.pyplot as plt
from cleantext import clean
from wordcloud import WordCloud, STOPWORDS
from textwrap import wrap
from collections import Counter
from collections import defaultdict
from textblob import TextBlob
import spacy


# =================================================================================
nltk.download('stopwords')
nltk.download('punkt')
stop_words = nltk.corpus.stopwords.words('english')
nlp = spacy.load("en_core_web_sm")
pd.set_option('display.expand_frame_repr', False)

# =================================================================================
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
BATCH_SIZE = 64
MAX_VOCAB_SIZE = 25_000
N_EPOCHS = 5
best_valid_loss = float('inf')
'''
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
submission.columns = submission.columns.str.replace('selftext', 'body')
comments = pd.concat([com1, com2], ignore_index=True)
# =================================================================================

## drop removed and empty post
submission = submission.drop(submission[(submission['body'] == '[removed]') | (submission['body'] == '')].index)
### drop nan
submission = submission.dropna()
'''

submission = pd.read_csv(r'../Data/submissions_tokenized.csv', encoding='utf-8')
comments = pd.read_csv(r'../Data/comments_tokenized.csv', encoding='utf-8')


## count
submission["sent_len"] = [len(sent_tokenize(x)) for x in submission['body']]
submission["word_count"] = [len(word_tokenize(x)) for x in submission['body']]
submission["word_avg_len"] = [sum(map(len, x.split())) / len(x.split()) for x in submission['body']]

plt.figure()
submission["sent_len"].hist(bins=100)
plt.title("Histogram of sentence length in Submissions")
plt.xlabel("Sentence Length")
plt.ylabel("Frequency")
plt.show()
plt.show()

plt.figure()
submission["word_count"].hist(bins=100)
plt.title("Histogram of word count in Submissions")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()

submission['body'] = submission['body'].str.lower()

# ### remove emoji
# submission = submission.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
# ### lower body text
# #### remove punctuation
# submission['body'] = submission['body'].str.replace(r'[-.{}/?!~,":;()\']', '')

## comments

comments = comments.drop(comments[(comments['body'] == '[removed]') | (comments['body'] == ' ')].index)
### drop nan
comments = comments.dropna()

comments["sent_len"] = [len(sent_tokenize(x)) for x in comments['body']]
comments["word_count"] = [len(word_tokenize(x)) for x in comments['body']]
comments["word_avg_len"] = [sum(map(len, x.split())) / len(x.split()) for x in comments['body']]

plt.figure()
comments["sent_len"].hist(bins=100)
plt.title("Histogram of sentence length in Commments")
plt.xlabel("Sentence Length")
plt.ylabel("Frequency")
plt.show()

plt.figure()
comments["word_count"].hist(bins=100)
plt.title("Histogram of word count in Comments")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()

### average word length in each sentence
plt.figure()
comments["word_avg_len"].hist(bins=100)
plt.title("Histogram of word average length in Comments")
plt.xlabel("Word Average Length")
plt.ylabel("Frequency")
plt.show()

comments['body'] = comments['body'].str.lower()
# =================================================================================

target = "subreddit"

### Submission
print("Submission", submission.groupby(target)[target].count())

# percent of target="MacOS"
target_count = submission[submission[target] == "MacOS"][target].count()
total = submission[target].count()
print("Percentage of data showing real incidents \"target=MacOS\" in Submission {0:.2f}".format(target_count / total))

target_count = submission[submission[target] == "windows"][target].count()
total = submission[target].count()
print("Percentage of data showing real incidents \"target=windows\" in Submission {0:.2f}".format(target_count / total))


###Comments
print("Comments", comments.groupby(target)[target].count())

# percent of target="MacOS"
target_count = comments[comments[target] == "MacOS"][target].count()
total = comments[target].count()
print("Percentage of data showing real incidents \"target=MacOS\" in Comments {0:.2f}".format(target_count / total))

target_count = comments[comments[target] == "windows"][target].count()
total = comments[target].count()
print("Percentage of data showing real incidents \"target=windows\" in Comments {0:.2f}".format(target_count / total))

# # histogram
# plt.figure()
# comments[comments[target] == "MacOS"][target].hist(label='MacOS', grid=False, bins=1, rwidth=0.8)
# comments[comments[target] == "windows"][target].hist(label='Windows', grid=False, bins=1, rwidth=0.8)
# plt.title('Histogram of MacOS/Windows in comments')
# plt.ylabel('Frequency')
# plt.xlabel('Subreddit')
# plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
sns.countplot(ax=axes[0], data=submission, x="subreddit")
sns.countplot(ax=axes[1], data=comments, x="subreddit")
axes[0].set_title('Histogram of MacOS/Windows in Submission')
axes[1].set_title('Histogram of MacOS/Windows in Comments')
plt.tight_layout()
plt.show()
# =================================================================================

corpus = []
new = submission[submission[target] == "MacOS"]['body'].str.split()
corpus = [word for i in new for word in i]

dic = defaultdict(int)
for word in corpus:
    if word in stop_words:
        dic[word] += 1

counter = Counter(corpus)
most = counter.most_common()
nltk.download('averaged_perceptron_tagger')
x, y = [], []
for word, count in most[:80]:
    if ((word not in stop_words) and ((nltk.pos_tag([word])[0][1] == "NN")  or (nltk.pos_tag([word])[0][1] == 'VB'))):
        x.append(word)
        y.append(count)

plt.figure(figsize=(12, 8))
sns.barplot(x=y, y=x)
plt.title("The most occurrences of word in Submissions related to MacOS")
plt.xlabel('Occurrences')
plt.ylabel('Words')
plt.show()

corpus = []
new = submission[submission[target] == "windows"]['body'].str.split()
corpus = [word for i in new for word in i]

dic = defaultdict(int)
for word in corpus:
    if word in stop_words:
        dic[word] += 1

counter = Counter(corpus)
most = counter.most_common()

x, y = [], []
for word, count in most[:80]:
    if ((word not in stop_words) and ((nltk.pos_tag([word])[0][1] == "NN")  or (nltk.pos_tag([word])[0][1] == 'VB'))):
        x.append(word)
        y.append(count)

plt.figure(figsize=(12, 8))
sns.barplot(x=y, y=x)
plt.title("The most occurrences of word in Submissions related to Windows")
plt.xlabel('Occurrences')
plt.ylabel('Words')
plt.show()

corpus = []
new = comments[comments[target] == "MacOS"]['body'].str.split()

corpus = [word for i in new for word in i]

dic = defaultdict(int)
for word in corpus:
    if word in stop_words:
        dic[word] += 1

counter = Counter(corpus)
most = counter.most_common()

x, y = [], []
for word, count in most[:80]:
    if ((word not in stop_words) and ((nltk.pos_tag([word])[0][1] == "NN")  or (nltk.pos_tag([word])[0][1] == 'VB'))):
        x.append(word)
        y.append(count)

plt.figure(figsize=(12, 8))
sns.barplot(x=y, y=x)
plt.title("The most occurrences of word in Comments related to MacOS")
plt.xlabel('Occurrences')
plt.ylabel('Words')
plt.show()

corpus = []
new = comments[comments[target] == "windows"]['body'].str.split()

corpus = [word for i in new for word in i]

dic = defaultdict(int)
for word in corpus:
    if word in stop_words:
        dic[word] += 1

counter = Counter(corpus)
most = counter.most_common()

x, y = [], []
for word, count in most[:80]:
    if ((word not in stop_words) and ((nltk.pos_tag([word])[0][1] == "NN")  or (nltk.pos_tag([word])[0][1] == 'VB'))):
        x.append(word)
        y.append(count)

plt.figure(figsize=(12, 8))
sns.barplot(x=y, y=x)
plt.title("The most occurrences of word in Comments related to Windows")
plt.xlabel('Occurrences')
plt.ylabel('Words')
plt.show()

# function for drawing a cloud of frequently used words


# Function for generating word clouds

from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)


def show_wordcloud(data, subreddit, post):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)

    wordcloud = wordcloud.generate(str(data))

    plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    plt.title(f"Word Cloud of {subreddit} in {post}")
    plt.imshow(wordcloud)
    plt.show()


show_wordcloud(str(submission[submission['subreddit'] == "MacOS"]['body']), "MacOS", "Submissions")
show_wordcloud(str(submission[submission['subreddit'] == "windows"]['body']), "Windows", "Submissions")

show_wordcloud(str(comments[comments['subreddit'] == "MacOS"]['body']), "MacOS", "Comments")
show_wordcloud(str(comments[comments['subreddit'] == "windows"]['body']), "Windows", "Comments")

'''
# =================================================================================
# Sentiment analysis

def polarity(text):
    return TextBlob(text).sentiment.polarity


submission['polarity_score'] = submission['body'].apply(lambda x: polarity(x))
comments['polarity_score'] = comments['body'].apply(lambda x: polarity(x))

plt.figure(figsize=(8, 8))
submission['polarity_score'][submission['subreddit'] == "MacOS"].hist()
plt.title('Sentiment analysis for MacOS in Submission')
plt.xlabel('polarity score')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 8))
submission['polarity_score'][submission['subreddit'] == "windows"].hist()
plt.title('Sentiment analysis for Windows in Submission')
plt.xlabel('polarity score')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 8))
comments['polarity_score'][comments['subreddit'] == "MacOS"].hist()
plt.title('Sentiment analysis for MacOS in comments')
plt.xlabel('polarity score')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 8))
comments['polarity_score'][comments['subreddit'] == "windows"].hist()
plt.title('Sentiment analysis for Windows in comments')
plt.xlabel('polarity score')
plt.ylabel('Frequency')
plt.show()


def sentiment(x):
    if x < 0:
        return 'Negative'
    elif x == 0:
        return 'Neutral'
    else:
        return 'Positive'


submission['polarity'] = submission['polarity_score'].map(lambda x: sentiment(x))
comments['polarity'] = comments['polarity_score'].map(lambda x: sentiment(x))

plt.figure(figsize=(8, 8))
plt.bar(submission['polarity'][submission['subreddit'] == "MacOS"].value_counts().index,
        submission['polarity'][submission['subreddit'] == "MacOS"].value_counts())
plt.title('Sentiment analysis for MacOS in Submission')
plt.xlabel('polarity')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 8))
plt.bar(submission['polarity'][submission['subreddit'] == "windows"].value_counts().index,
        submission['polarity'][submission['subreddit'] == "windows"].value_counts())
plt.title('Sentiment analysis for Windows in Submission')
plt.xlabel('polarity')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 8))
plt.bar(comments['polarity'][comments['subreddit'] == "MacOS"].value_counts().index,
        comments['polarity'][comments['subreddit'] == "MacOS"].value_counts())
plt.title('Sentiment analysis for MacOS in comments')
plt.xlabel('polarity')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 8))
plt.bar(comments['polarity'][comments['subreddit'] == "windows"].value_counts().index,
        comments['polarity'][comments['subreddit'] == "windows"].value_counts())
plt.title('Sentiment analysis for Windows in comments')
plt.xlabel('polarity')
plt.ylabel('Frequency')
plt.show()

######## Sentiment vs. Post score
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
sns.lineplot(ax= axes[0, 0], data=submission[submission['subreddit'] == "MacOS"], x="polarity_score", y="score", hue="subreddit")
sns.lineplot(ax= axes[0, 1], data=submission[submission['subreddit'] == "windows"], x="polarity_score", y="score", hue="subreddit")
sns.lineplot(ax= axes[1, 0], data=comments[comments['subreddit'] == "MacOS"], x="polarity_score", y="score", hue="subreddit")
sns.lineplot(ax= axes[1, 1],data=comments[comments['subreddit'] == "windows"], x="polarity_score", y="score", hue="subreddit")
plt.show()
#
# plt.figure(figsize=(8, 8))
# sns.lineplot(data=submission[submission['subreddit'] == "MacOS"], x="polarity_score", y="score", hue="subreddit")
# plt.ylabel('score')
# plt.xlabel('polarity score')
# plt.title('Sentiment vs. Post score for MacOS in Submission')
# plt.show()
#
# plt.figure(figsize=(8, 8))
# sns.lineplot(data=submission[submission['subreddit'] == "windows"], x="polarity_score", y="score", hue="subreddit")
# plt.ylabel('score')
# plt.xlabel('polarity score')
# plt.title('Sentiment vs. Post score for Windows in Submission')
# plt.show()
#
# plt.figure(figsize=(8, 8))
# sns.lineplot(data=comments[comments['subreddit'] == "MacOS"], x="polarity_score", y="score", hue="subreddit")
# plt.ylabel('score')
# plt.xlabel('polarity score')
# plt.title('Sentiment vs. Post score for MacOS in comments')
# plt.show()
#
# plt.figure(figsize=(8, 8))
# sns.lineplot(data=comments[comments['subreddit'] == "windows"], x="polarity_score", y="score", hue="subreddit")
# plt.ylabel('score')
# plt.xlabel('polarity score')
# plt.title('Sentiment vs. Post score for Windows in comments')
# plt.show()


# another way of sentiment analysis but i think it does not work well.

# import numpy as np
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
#
# nltk.download('vader_lexicon')
# sid = SentimentIntensityAnalyzer()
#
# def get_vader_score(sent):
#     ss = sid.polarity_scores(sent)
#     return np.argmax(list(ss.values())[:-1])
#
# submission['polarity_v']= submission['body'].map(lambda x: get_vader_score(x))
# sub_polarity = submission['polarity_v'].replace({0:'neg',1:'neu',2:'pos'})
#
# plt.figure()
# plt.bar(sub_polarity.value_counts().index, sub_polarity.value_counts())
# plt.show()

# =================================================================================
#### Named entity recognition


def ner(text):
    doc = nlp(text)
    return [X.label_ for X in doc.ents]


submission['ent'] = submission['body'].apply(lambda x: ner(x))
count_sub_win = Counter([x for sub in submission['ent'][submission['subreddit'] == "windows"] for x in sub]).most_common()
count_sub_mac = Counter([x for sub in submission['ent'][submission['subreddit'] == "MacOS"] for x in sub]).most_common()
comments['ent'] = comments['body'].apply(lambda x: ner(x))
count_com_win = Counter([x for sub in comments['ent'][comments['subreddit'] == "windows"] for x in sub]).most_common()
count_com_mac = Counter([x for sub in comments['ent'][comments['subreddit'] == "MacOS"] for x in sub]).most_common()

x_sub_win, y_sub_win = map(list, zip(*count_sub_win))
x_sub_mac, y_sub_mac = map(list, zip(*count_sub_mac))
x_com_win, y_com_win = map(list, zip(*count_com_win))
x_com_mac, y_com_mac = map(list, zip(*count_com_mac))
plt.figure(figsize=(8, 8))
sns.barplot(x=y_sub_win, y=x_sub_win)
sns.barplot(x=y_sub_mac, y=x_sub_mac)
sns.barplot(x=y_com_win, y=x_com_win)
sns.barplot(x=y_com_mac, y=x_com_mac)
plt.show()
'''