
## Classify sentiment for full sample
## Author: Joan Timoneda
## Date: November 25 2025


from openai import OpenAI 
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import os 

os.chdir(os.path.dirname(os.path.abspath(__file__)))

client = OpenAI(api_key="REPLACE_WITH_YOUR_OPENAI_API_KEY")

# Read in overall data

data = pd.read_excel("all_sentiment_true.xlsx", index_col=0) # use 
data.rename(columns={"tweet":"text"}, inplace=True)
data["sentiment"] = data["sentiment"].str.lower()
data = data[["text", "sentiment", "label"]]
data.sentiment.value_counts()
data

# Draw few-shot learning examples: 

#Examples
neg = data[(data["label"] == 0)].sample(5, random_state=10)
pos = data[(data["label"] == 1)].sample(5, random_state=10)

shots = pd.concat([neg, pos]).sample(frac=1, random_state=10).reset_index(drop=True)
few_shot_examples = ""
for i, row in shots.iterrows():
    few_shot_examples += (
        f"Example {i+1}:\n"
        f"Text: {row['text']}\n"
        f"Label: {row['sentiment']}\n"
        f"Numeric Label: {row['label']}\n\n")
few_shot_examples
print(few_shot_examples)


## BEGIN ANNOTATION: 

system_prompt = f"""
You will classify the sentiment of short political social-media posts.

Label definitions:
- POSITIVE (1): the post expresses supportive, approving, optimistic, or favorable sentiment.
- NEGATIVE (0): the post expresses critical, unfavorable, disapproving, or pessimistic sentiment.

Here are 10 labeled examples:
{few_shot_examples}

Your task:
Given a NEW POST, respond ONLY with "0" or "1". No explanations or extra text.
"""
print(system_prompt)


# Classification function
def classify(system_prompt, text):
    
    user_message = f"Classify the following political social-media post.\n\nText:\n{text}\n\nRespond only with 0 or 1."
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0,
        max_tokens=5
    )
    
    return response.choices[0].message.content.strip()


# Apply
text = data.text

# Run 1:
response = [classify(system_prompt, x) for x in tqdm(text)]
print(response)

preds = pd.Series([int(p) for p in response])
preds.value_counts()

all_synth = pd.concat([data, preds.rename("preds")], axis=1)
all_synth

all_synth.to_excel("data/classified/classified_full_true_sample.xlsx")
















