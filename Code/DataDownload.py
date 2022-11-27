# Standard Packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# Web Request Package
import requests

'''
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
    base_url = f"https://api.pushshift.io/reddit/search/{data_type}/?subreddit={q}&sort=asc"
    payload = kwargs
    headers = {'User-agent': 'eamonious'}
    request = requests.get(base_url, params=payload, headers=headers)
    json = request.json()
    data = pd.DataFrame(json['data'])
    return data

### submission
submission = get_pushshift_data("submission",
                                q="Android",
                                size=500)
len_sub = len(submission)

'''
from pmaw import PushshiftAPI
import datetime as dt

def get_pushshift(subreddit, limit, before, after, filter_attr, datat_type):
    api = PushshiftAPI()
    if datat_type =='submission':
        submission_result= api.search_submissions(subreddit=subreddit, limit=limit, before=before, after=after, mem_safe=True, filter=filter_attr)
        print(f'Retrieved {len(submission_result)} posts from Pushshift')
        submission_df = pd.DataFrame(submission_result)
        return submission_df
    if datat_type == 'comment':
        # return the comments that match the submission id
        comment_result = api.search_comments(subreddit=subreddit, limit=limit, before=before, after=after, mem_safe=True)
        print(f'Retrieved {len(comment_result)} comments from Pushshift')
        comment_df = pd.DataFrame(comment_result)
        return comment_df
# cite: https://pypi.org/project/pmaw/
# https://medium.com/swlh/how-to-scrape-large-amounts-of-reddit-data-using-pushshift-1d33bde9286

limit = 40000
before = int(dt.datetime(2022,10,30,0,0).timestamp())
after = int(dt.datetime(2021,1,1,0,0).timestamp())

subreddit="MacOS"
# Removes everything but the features we are interested in.
filter_attr = ['id', "author", 'is_self', 'is_video', 'subreddit', 'link_flair_text',
                         'num_comments', 'over_18', 'post_hint', 'score', 'title', 'selftext', "subreddit"]

print('Start download submissions')
submissions = get_pushshift(subreddit, limit, before, after, filter_attr, datat_type='submission')
submissions.to_csv(f'./{subreddit}_submission.csv', header=True, index=False, columns=list(submissions.axes[1]), encoding='utf-8')
print(f'Total submissions has {submissions.shape}')

print('Start download comments')
limit2 = 400000
filter_attr2 = ['id', 'link_id', 'author', 'subreddit', 'score', 'body']
comments = get_pushshift(subreddit, limit2, before, after, filter_attr2, datat_type='comment')
comments.to_csv(f'./{subreddit}_comments.csv', header=True, index=False, columns=list(comments.axes[1]), encoding='utf-8')
print(f'Total comments has {comments.shape}')

# ## drop removed and empty post
# submission_iphone.drop(submission_iphone[(submission_iphone['selftext'] == '[removed]') | (submission_iphone['selftext'] == '')].index,
#                 inplace=True)
#
# ### comments
# # Removes everything but the features we are interested in.
# comment = comment[['author_fullname', 'created_utc', 'id', 'is_submitter', 'subreddit',
#                    'send_replies', 'subreddit_type', 'score', 'body']]
#
# ## drop removed and empty post
# comment.drop(comment[(comment['body'] == '[removed]') | (comment['body'] == '')].index, inplace=True)
