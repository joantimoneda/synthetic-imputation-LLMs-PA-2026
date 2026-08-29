
from openai import OpenAI 
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import os 
import time

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))  # path anchored for replication package

client = OpenAI(api_key="REPLACE_WITH_YOUR_OPENAI_API_KEY")

#run = 2 ## SET RUN EARLY AND FIRST OFF. THIS WILL BE USED FOR DEFAULT SAVING LATER (IF OUTSIDE LOOP)

# models = client.models.list()
# for m in models.data:
#     print(m.id) ## To see the full list of available models and their names



# Read in data FOR EXAMPLES, same as with full sample
data = pd.read_excel("../../nostalgia_data_clean_sample.xlsx", index_col=0) # use 
len(data)

#Examples
nost = data[data["nostalgic"] == 1].sample(5, random_state=10)
non_nost = data[data["nostalgic"] == 0].sample(5, random_state=10)
shots = pd.concat([nost, non_nost]).sample(frac=1, random_state=123).reset_index(drop=True)
few_shot_examples = ""
for i, row in shots.iterrows():
    few_shot_examples += (
        f"Example {i+1}:\n"
        f"Text: {row['text']}\n"
        f"Label: {row['nostalgic']}\n\n"
    )
few_shot_examples
print(few_shot_examples)


# Define the system prompt
prompt = f"""
You are a text classification model. Your task is to classify short political texts as:

0 = Not nostalgic  
1 = Nostalgic

You will see 10 labeled examples. Study them carefully and follow the same labeling logic.  
Return ONLY the numeric label (0 or 1). Do not explain your answer.

Here are the examples:

{few_shot_examples}
"""
print(prompt)

## SYNTHETIC DATA TO CLASSIFY
synth = pd.read_excel("../data synthetic/synthetic_100_51.xlsx", index_col=0) # use 
len(synth)


# Classification function
def classify(system_prompt, text):
    
    user_message = f"Classify the following text.\n\nText:\n{text}\n\nReturn only 0 or 1."
    
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
text = synth.text

# Run 1:
response = [classify(prompt, x) for x in tqdm(text)]
print(response)

response = pd.Series(response, name="preds").astype(int)
print(response)
response.value_counts()

all_synth = pd.concat([synth, response], axis=1)
all_synth

all_synth.to_excel("GPT_nost_synth_51.xlsx")




