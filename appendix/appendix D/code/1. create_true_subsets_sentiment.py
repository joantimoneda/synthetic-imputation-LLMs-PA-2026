
## Create subsamples 
## Author: Joan Timoneda
## Date: November 21 2025

import re
import os
import pandas as pd
import numpy as np

os.chdir("/Users/joan/Dropbox/3 Art. under Review/Synthetic imputation NLP/sentiment")

d = pd.read_excel("data/all_sentiment_true.xlsx", index_col=0)
d
d.label.value_counts()

# subset a bit for some balance
pos = d[d["label"] == 1]
neg = d[d["label"] == 0].sample(n=len(pos), random_state=10)

d = pd.concat([pos, neg], ignore_index=True)
d = d.sample(frac=1, random_state=10).reset_index(drop=True) 

d.label.value_counts()

############################

# First we generate subsamples for 50, 75, 100, and 150


## Ok, now create the 50 sample from this sample: 

true_pos_50 = d[d.label == 1].sample(n=50, random_state=10).reset_index(drop=True)
true_50 = pd.concat([true_pos_50, d[d["label"] == 0]], ignore_index=True)
true_50 = true_50.sample(frac=1, random_state=10).reset_index(drop=True)
true_50.label.value_counts()

true_50.to_excel("data/true_samples/true_50_sample.xlsx")


## Now create the 75 sample

true_pos_75 = d[d.label == 1].sample(n=75, random_state=10).reset_index(drop=True)
true_75 = pd.concat([true_pos_75, d[d["label"] == 0]], ignore_index=True)
true_75 = true_75.sample(frac=1, random_state=10).reset_index(drop=True)
true_75.label.value_counts()

true_75.to_excel("data/true_samples/true_75_sample.xlsx")

## Now create the 100 sample

true_pos_100 = d[d.label == 1].sample(n=100, random_state=10).reset_index(drop=True)
true_100 = pd.concat([true_pos_100, d[d["label"] == 0]], ignore_index=True)
true_100 = true_100.sample(frac=1, random_state=10).reset_index(drop=True)
true_100.label.value_counts()

true_100.to_excel("data/true_samples/true_100_sample.xlsx")


## Now create the 150 sample

true_pos_150 = d[d.label == 1].sample(n=150, random_state=10).reset_index(drop=True)
true_150 = pd.concat([true_pos_150, d[d["label"] == 0]], ignore_index=True)
true_150 = true_150.sample(frac=1, random_state=10).reset_index(drop=True)
true_150.label.value_counts()

true_150.to_excel("data/true_samples/true_150_sample.xlsx")


## Create a 250 sample: 

true_pos_250 = d[d.label == 1].sample(n=250, random_state=10).reset_index(drop=True)
true_250 = pd.concat([true_pos_250, d[d["label"] == 0]], ignore_index=True)
true_250 = true_250.sample(frac=1, random_state=10).reset_index(drop=True)
true_250.label.value_counts()

true_250.to_excel("data/true_samples/true_250_sample.xlsx")














