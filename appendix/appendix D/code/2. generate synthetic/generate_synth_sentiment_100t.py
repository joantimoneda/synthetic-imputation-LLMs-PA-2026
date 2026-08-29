
## Generate synth for 100 true samples
## Author: Joan Timoneda
## Date: November 21 2025

import os
from openai import OpenAI 
import pandas as pd
from tqdm import trange

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# API Key:
client = OpenAI(api_key="REPLACE_WITH_YOUR_OPENAI_API_KEY")

# Get the data
d = pd.read_excel("true_100_sample.xlsx", index_col=0)
d = d[d["label"]==1].reset_index(drop=True)
d = d.drop(columns=["stance", "opinion"])
d.rename(columns={"tweet":"text"}, inplace=True)
d.label.value_counts()
d

prompt = """Generate a tweet in English based on the examples below.\n\
Make sure the new text is different in structure, word choice and content from the examples, but still retains the underlying positive tone. Names, countries and topics should also be different. The new text should also be no more than one or two full sentences, and have similar length to the examples. Here are the five examples:\n\
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
for i in trange(0, 100): 
    examples = d["text"].sample(n=5).reset_index(drop=True)
    ex1, ex2, ex3, ex4, ex5 = examples
    prompt_with_examples = prompt.format(ex1, ex2, ex3, ex4, ex5)
    r.append(input(prompt_with_examples, 0.7))
print(r)
r
len(r)

exp = pd.concat([pd.Series(r).rename("text"), pd.Series([1]*len(r)).rename("synthetic")], axis=1)
exp.to_excel("data/synthetic/synthetic_100t_100.xlsx")
