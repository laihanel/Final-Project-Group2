# Standard Packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# Web Request Package
import requests


# # The url given below calls for the most recent 1000 comments from threads on r/AskMen.
# url = "https://api.pushshift.io/reddit/search/submission/?subreddit=Windows&sort=des&size=1000"
#
# headers = {'User-agent': 'eamonious'}
# res = requests.get(url, headers=headers)
# res.status_code
#
# json = res.json()
# submission = pd.DataFrame(json['data'])
# submission.columns


def get_pushshift_data(data_type, q, **kwargs):
    base_url = f"https://api.pushshift.io/reddit/search/{data_type}/?q={q}/"
    payload = kwargs
    headers = {'User-agent': 'eamonious'}
    request = requests.get(base_url, params=payload, headers=headers)
    json = request.json()
    data = pd.DataFrame(json['data'])
    return data

### submission
submission = get_pushshift_data("submission",
                                q="iphone",
                                size=1000)
len_sub = len(submission)

# Removes everything but the features we are interested in.
submission = submission[['author_fullname', 'created_utc', 'id', 'is_self', 'is_video', 'subreddit', 'link_flair_text',
                         'num_comments', 'over_18', 'post_hint', 'score', 'title', 'selftext']]

## drop removed and empty post
submission.drop(submission[(submission['selftext'] == '[removed]') | (submission['selftext'] == '')].index,
                inplace=True)


### comments
comment = get_pushshift_data("comment",
                             q="iphone",
                             size=1000)
len_com = len(comment)

# Removes everything but the features we are interested in.
comment = comment[['author_fullname', 'created_utc', 'id', 'is_submitter', 'subreddit',
                   'send_replies', 'subreddit_type', 'score', 'body']]

## drop removed and empty post
comment.drop(comment[(comment['body'] == '[removed]') | (comment['body'] == '')].index, inplace=True)
