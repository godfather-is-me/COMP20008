import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot

from nltk.corpus import stopwords
import nltk
# nltk.download('punkt')
# nltk.downlaod('stopwords')

from itertools import combinations
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans

amz = pd.read_csv("amazon.csv")
gg = pd.read_csv("google.csv")

amz.drop(['manufacturer', 'description'], axis=1, inplace=True)
gg.drop(['manufacturer', 'description'], axis=1, inplace=True)
stop_words = set(stopwords.words('english'))

idAmz = []
idGG = []
amzBlk = []
ggBlk = []

def wordBigram(title):
    tokens = title.split()
    words = [w for w in tokens if not w in stop_words and len(w)>3 and not w.isnumeric()]
    bigram = []
    if words:
        bigram = list(combinations(words, 2))
    return bigram

count = 0
for title in amz['title']:
    bigram = wordBigram(title)
    if bigram:
        for gram in bigram:
            idAmz.append(amz['idAmazon'][count])
            amzBlk.append(gram)
    count += 1

count = 0
for title in gg['name']:
    bigram = wordBigram(title)
    if bigram:
        for gram in bigram:
            idGG.append(gg['id'][count])
            ggBlk.append(gram)
    count += 1

amz_csv = pd.DataFrame({'block_key':amzBlk, 'product_id': idAmz})
gg_csv = pd.DataFrame({'block_key': ggBlk, 'product_id': idGG})

amz_csv.to_csv("amazon_blocks.csv", index=False)
gg_csv.to_csv("google_blocks.csv", index=False)