from transformers import pipeline
from transformers import BertTokenizer, AutoTokenizer, BertModel, BertConfig, AutoModel, AdamW
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from scipy.special import softmax
import csv
import urllib.request
# sentiment_model = pipeline(model="laihanel/sentiment-analysis_gwu")
# sentiment_model(["I love this move", "This movie sucks!"])

from transformers import pipeline

# Set up the inference pipeline using a model from the 🤗 Hub
# sentiment_analysis = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
sentiment_analysis = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")



sub1 = pd.read_csv(r'../Data/MacOS_submission.csv', encoding='utf-8')
sub2 = pd.read_csv(r'../Data/windows_submission.csv', encoding='utf-8')

submission = pd.concat([sub1, sub2],ignore_index=True)
submission['body'] = submission['title'].astype(str) + ' ' + submission['selftext'].astype(str)
for text in submission['body'][1:10]:
    # text = "Good night 😊"
    print(f"{str(text)}")
    encoded_input = tokenizer(text, return_tensors='pt')
    output = sentiment_analysis(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]
        print(f"{i+1}) {l} {np.round(float(s), 4)}")