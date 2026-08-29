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
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


# API Key JOAN:
client = OpenAI(api_key="REPLACE_WITH_YOUR_OPENAI_API_KEY")

# Directory
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # path anchored for replication package

# Load  data:
data = pd.read_excel("data samples/real_nost_100.xlsx", index_col=0)
data
data.text

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
    
data.text.isna().value_counts()

## Loop through to get results
r = []
for i in trange(0,51): # 151-50
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
exp.text.apply(type).value_counts()
# Are there empty or whitespace-only strings?
print("Empty strings:", (exp["text"].str.strip().str.len() == 0).sum())
# Are there strings starting with characters Excel will mangle?
print("Excel-dangerous starts:", exp["text"].str.match(r"^[=+\-@]").sum())

exp.to_excel("data synthetic/synthetic_100_51.xlsx")


