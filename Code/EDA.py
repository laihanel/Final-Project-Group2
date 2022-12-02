@@ -0,0 +1,64 @@
import pandas as pd
import nltk
import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator, MonthLocator
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from datetime import datetime

# read data
submissions_windows = pd.read_csv(r'../Data/windows_submission.csv', encoding='utf-8')
submissions_mac = pd.read_csv(r'../Data/MacOS_submission.csv', encoding='utf-8')
 # stack the two DataFrames
submissions = pd.concat([submissions_windows, submissions_mac], ignore_index=True, axis=0)
print(submissions.shape)
# 68595 total
print(submissions.head(10))

# comments_windows = pd.read_csv(r'./Data/windows_comments.csv', encoding='utf-8')
# comments_mac = pd.read_csv(r'./Data/MacOS_comments.csv', encoding='utf-8')
# # stack the two DataFrames
# comments = pd.concat([comments_windows, comments_mac], ignore_index=True, axis=0)
# print(comments.shape)
# print(comments.head(10))

# remove value has [removed] or [deleted] in text body in submission
submissions.drop(submissions[(submissions['selftext'] == '[removed]') | (submissions['selftext'] == '[deleted]')].index, inplace=True)
print(submissions.shape)
# 53865 data remain

#
sns.countplot(data=submissions, x='subreddit',)
plt.title('Count of Submissions')
plt.show()


# change the datetype
submissions.sort_values(by='created_utc')
submissions['created_year'] = pd.to_datetime(submissions['created_utc'], unit='s').dt.year
submissions['created_mon'] = pd.to_datetime(submissions['created_utc'], unit='s').dt.month
submissions['created_week'] = pd.to_datetime(submissions['created_utc'], unit='s').dt.isocalendar().week

submissions['created_week'] = np.where((submissions['created_mon'] == 1) & (submissions['created_week'] >= 52), 0, submissions['created_week'])

subcount = submissions.groupby(['created_year', 'created_mon', 'created_week', 'subreddit'])['id'].count().reset_index(name='counts')
subcount['Date'] = subcount['created_year'].astype(str) + '-' + subcount['created_mon'].astype(str) + '-' + subcount['created_week'].astype(str)
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x='Date',
             y='counts',
             hue='subreddit',
             data=subcount).set(ylabel='submissions', title="Number of submission by week")

# ax.xaxis.set_major_locator(MonthLocator())
# ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
ax.grid(axis='y')
ax.set_xticklabels(['2020', '2021-Jan', '2021-Apr', '2021-Jul', '2021-Oct', '2022-Jan', '2022-Apr', '2022-Jul', '2022-Oct'], rotation = 45)
ax.xaxis.set_major_locator(plt.MaxNLocator(9))
plt.tight_layout()
plt.show()


