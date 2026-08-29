##################################
## GPT-5.1 synthetic data
## Author: Joan C Timoneda
## Date: May 4, 2026
#################################

import os
from openai import OpenAI 
import pandas as pd
import numpy as np
import signal
import datetime
import time
from tqdm import tqdm, trange

# API Key JOAN:
client = OpenAI(api_key="REPLACE_WITH_YOUR_OPENAI_API_KEY")

# Directory
os.chdir("/Users/joan/Dropbox/2 Art. R&R/Synthetic imputation NLP/nostalgia/A_length_fix_RR")

# Load  data:
data = pd.read_excel("../data samples/real_nost_50.xlsx", index_col=0)
data

# Grab only 25
data = data.sample(n=25).reset_index(drop=True)
data

# For efficiency and savings, use 101 synthetic for 50 true run
synth_101 = pd.read_excel("data synthetic/synthetic_50_101.xlsx", index_col=0)
synth_101


# GPT 5:

prompt = """Generate a sentence in English based on the examples below. Make sure the new text has a similar nostalgic tone to the examples but is different in content, names, countries and topics. Match the mean length of the examples: {mean_length} words. Here are the five examples:\n\
Example 1: {ex1}\n\
Example 2: {ex2}\n\
Example 3: {ex3}\n\
Example 4: {ex4}\n\
Example 5: {ex5}"""


def input(prompt, temp, mean_length):
    response = client.chat.completions.create(
      model="gpt-5.1",
      messages=[{"role": "system", "content": prompt}],      
      temperature=temp,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
    )
    return response.choices[0].message.content.strip()
    


## Loop through to get results
r = []
for i in trange(0,25):
    examples = data["text"].sample(n=5).reset_index(drop=True)
    ex1, ex2, ex3, ex4, ex5 = examples
    mean_length = np.mean([len(x.split()) for x in examples])
    prompt_with_examples = prompt.format(ex1=ex1, ex2=ex2, ex3=ex3, ex4=ex4, ex5=ex5, mean_length=int(mean_length))
    r.append(input(prompt_with_examples, 0.7, mean_length))

r
len(r)

np.mean([len(x.split()) for x in r])

exp = pd.concat([pd.Series(r).rename("text"), pd.Series([1]*len(r)).rename("synthetic")], axis=1)
exp.text.isna().sum()

exp = pd.concat([synth_101, exp]).reset_index(drop=True)
exp

exp.to_excel("data synthetic/synthetic_25_126.xlsx")


# combine datasets

synth = pd.read_excel("data synthetic/synthetic_25_126.xlsx", index_col=0) # if not in memory
synth.text.apply(type).value_counts()
synth["nostalgic"] = 1

data
data["synthetic"] = 0

comb = pd.concat([data, synth], axis=0)
comb.synthetic.value_counts()

not_nost = pd.read_excel("../nostalgia_data_clean_sample.xlsx", index_col=0)
not_nost = not_nost[not_nost["nostalgic"]==0]
not_nost = not_nost.drop("train", axis=1)
not_nost["synthetic"] = 0

print("NaN in synth:", synth["text"].isna().sum())
print("NaN in data (real nostalgic):", data["text"].isna().sum())
print("NaN in not_nost:", not_nost["text"].isna().sum())
not_nost = not_nost.dropna(subset=["text"]).reset_index(drop=True)

final_data = pd.concat([comb, not_nost], axis=0)
final_data.nostalgic.value_counts()
final_data.synthetic.value_counts()
final_data = final_data.sample(frac=1).reset_index(drop=True)
final_data.text.apply(type).value_counts()

final_data.to_excel("data synthetic/synthetic_25_126_fullsample.xlsx")

final_data_reloaded = pd.read_excel("data synthetic/synthetic_25_126_fullsample.xlsx", index_col=0)
print("final_data after reload:", final_data_reloaded["text"].isna().sum())
