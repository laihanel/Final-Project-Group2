#Standard Packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

#Web Request Package
import requests

#The url given below calls for the most recent 1000 comments from threads on r/AskMen.
url = "https://api.pushshift.io/reddit/search/submission/?subreddit=Windows&sort=des&size=1000"

headers = {'User-agent': 'eamonious'}
res = requests.get(url, headers=headers)
res.status_code

json = res.json()
submission = pd.DataFrame(json['data'])
submission.columns

#Removes everything but the features we are interested in.
submission = submission[['author_fullname','created_utc','id','is_self','is_video','subreddit', 'link_flair_text',
                         'num_comments', 'over_18', 'post_hint', 'score', 'title', 'selftext']]