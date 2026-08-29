##################################
## GPT-5.1 synthetic data
## Author: Joan C Timoneda
## Date: Nov 19, 2025
#################################

import os
from openai import OpenAI 
import pandas as pd
from tqdm import trange

# API Key JOAN:
client = OpenAI(api_key="REPLACE_WITH_YOUR_OPENAI_API_KEY")

# Directory
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))  # path anchored for replication package

# Load  data:
data = pd.read_excel("intl_speeches_sample_100.xlsx", index_col=0)
data

prompt = """Generate a 500-word speech in English. Make sure the new text is one whole paragraph and is different in content, names and country from the following five examples:\n\
Example 1: {}\n\
Example 2: {}\n\
Example 3: {}\n\
Example 4: {}\n\
Example 5: {}"""
print(prompt)


def input(prompt, temp):
    response = client.chat.completions.create(
      model="gpt-5.1",
      messages=[
          {"role": "user", "content": prompt}
      ],
      temperature=temp,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
    )
    return response.choices[0].message.content.strip()


## Loop through to get results
r = []
for i in trange(0, 118): # 218-100
    examples = data["text_eng"].sample(n=5).reset_index(drop=True)
    ex1, ex2, ex3, ex4, ex5 = examples
    prompt_with_examples = prompt.format(ex1, ex2, ex3, ex4, ex5)
    r.append(input(prompt_with_examples, 0.8))
print(r)
r
len(r)

exp = pd.concat([pd.Series(r).rename("text"), pd.Series([1]*len(r)).rename("synthetic")], axis=1)
exp.to_excel("synthetic_100_118.xlsx")






