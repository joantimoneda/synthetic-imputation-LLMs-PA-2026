
import re
import os
import pandas as pd
import numpy as np

os.chdir("/Users/joan/Dropbox/3 Art. under Review/Synthetic imputation NLP/sentiment")

d = pd.read_csv("data/raw/sem_eval_2016_stance_sentiment_train.txt",
    sep="\t", header=0, 
    names=["id", "target", "tweet", "stance", "opinion", "sentiment"],
    quoting=3, encoding="utf-8",  on_bad_lines="skip")
d.head()
len(d)

d2 = pd.read_csv("data/raw/sem_eval_2016_stance_sentiment_test.txt",
    sep="\t", header=0, 
    names=["id", "target", "tweet", "stance", "opinion", "sentiment"],
    quoting=3, encoding="latin1",  on_bad_lines="skip")

d3 = pd.read_csv("data/raw/testdata-taskB-all-annotations.txt",
    sep="\t", header=0, 
    names=["id", "target", "tweet", "stance", "opinion", "sentiment"],
    quoting=3, encoding="latin1",  on_bad_lines="skip")

d4 = pd.read_csv("data/raw/trainingdata-all-annotations.txt",
    sep="\t", header=0, 
    names=["id", "target", "tweet", "stance", "opinion", "sentiment"],
    quoting=3, encoding="latin1",  on_bad_lines="skip")


all = pd.concat([d, d2, d3, d4], axis=0).reset_index(drop=True)
all

all = all.drop_duplicates()
all

all.sentiment.value_counts()

all["label"] = all["sentiment"].map({"NEGATIVE": 0, "POSITIVE": 1})
all = all.dropna().reset_index(drop=True)
all["label"] = all["label"].astype(int)
all = all.drop("id", axis=1, errors="ignore")
all

all.to_excel("data/all_sentiment_true.xlsx")