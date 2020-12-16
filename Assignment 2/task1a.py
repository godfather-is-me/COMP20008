import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import nltk

ams = pd.read_csv("amazon_small.csv")
ggs = pd.read_csv("google_small.csv")

# ------------------------ Scoring ---------------------------
# Max score is 100

ams2 = ams.copy()
ggs2 = ggs.copy()
# Since we drop values when required
ams_extra = []

#fuz, fuz, fuz, 1-(diff/100)
for indexA, rowA in ams2.iterrows():
    max_score = 0
    max_index = -1
    
    # print(rowA['title'])
    result = process.extractBests(rowA['title'], ggs2['name'], scorer=fuzz.partial_ratio, limit=1, score_cutoff=65)
    if result:
    #print(result[0])
        G_title = result[0][0]
        max_score = result[0][1]
        max_index = result [0][2]

        # Max value received
        ams_extra.append((indexA, max_index, max_score))

        # Confirmation
        # print((indexA , max_index, max_score))

        # Update records
        ams2.drop(indexA, inplace=True)
        ggs2.drop(max_index, inplace=True)
    
aID = []
gID = []
for ele in ams_extra:
    aID.append(ams.iloc[ele[0]]['idAmazon'])
    gID.append(ggs.iloc[ele[1]]['idGoogleBase'])
    
final = pd.DataFrame({'idAmazon':aID, 'idGoogleBase': gID})
final.to_csv("task1a.csv", index=False)