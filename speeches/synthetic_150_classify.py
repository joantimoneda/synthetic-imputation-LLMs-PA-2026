
from openai import OpenAI 
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import os 

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))  # path anchored for replication package

client = OpenAI(api_key="REPLACE_WITH_YOUR_OPENAI_API_KEY")

# Data to combine
data_annotations = pd.read_excel("../GPT_speeches_annotations.xlsx", index_col=0) # use 
len(data_annotations)
data_annotations
not_intl = data_annotations[data_annotations["label"] != 1].reset_index(drop=True)

synth_new = pd.read_excel("synthetic_150_68.xlsx", index_col=0) 
synth_new["speechtype"] = "international"
synth_new["label"] = 1
synth_new = synth_new.sample(n=68).reset_index(drop=True)

other_cats = not_intl.drop(["text", "preds"], axis=1)
other_cats.rename(columns={"text_eng": "text"}, inplace=True)
other_cats["synthetic"] = 0

real100 = pd.read_excel("../../data samples/intl_speeches_sample_150.xlsx", index_col=0) # use 
real100 = real100[["text_eng", "speechtype", "label"]]
real100.rename(columns={"text_eng": "text"}, inplace=True)
real100["label"] = 1
real100["synthetic"] = 0

# Put final data together to classify
data = pd.concat([synth_new, other_cats, real100]).reset_index(drop=True)
data.speechtype.value_counts()
data.label.value_counts()
data.synthetic.value_counts()
data = data.sample(frac=1).reset_index(drop=True)

#Examples
fam = data[data["speechtype"] == "famous"].sample(5, random_state=10)
int = data[data["speechtype"] == "international"].sample(5, random_state=10)
rib = data[data["speechtype"] == "ribboncutting"].sample(5, random_state=10)
cam = data[data["speechtype"] == "campaign"].sample(5, random_state=10)

shots = pd.concat([fam, int, rib, cam]).sample(frac=1, random_state=10).reset_index(drop=True)
few_shot_examples = ""
for i, row in shots.iterrows():
    few_shot_examples += (
        f"Example {i+1}:\n"
        f"Text: {row['text']}\n"
        f"Label: {row['speechtype']}\n"
        f"Numeric Label: {row['label']}\n\n")
few_shot_examples
print(few_shot_examples)


# Define the system prompt
prompt = f"""
You are a text classification model. Your task is to classify political speeches into one of the following four categories:

0 = Famous speech
    Definition: A historically well-known or widely cited speech written for a general audience, often addressing national identity, values, or major moral principles. Not tied to a specific event, ceremony, or campaign rally.

1 = International speech
    Definition: A speech delivered to a foreign audience, international institution, diplomatic gathering, or global forum. It references foreign policy, international cooperation, global issues, or relations between nations.

2 = Ribbon-cutting speech
    Definition: A speech linked to the inauguration or opening of a public building, infrastructure, local project, school, hospital, bridge, cultural center, etc. It focuses on celebrating a new public good, thanking local actors, or marking the completion of a concrete project.

3 = Campaign speech
    Definition: A speech delivered during an electoral campaign. It refers to party platforms, political opponents, promises, mobilizing supporters, rallying the electorate, or encouraging people to vote.

Your task:
- Read the input text.
- Decide which ONE of the four categories fits best.
- Return ONLY the numeric label (0, 1, 2, or 3).
- Do NOT explain your answer.

Here are the examples:

{few_shot_examples}
"""
print(prompt)


# Classification function
def classify(system_prompt, text):
    
    user_message = f"Classify the following text.\n\nText:\n{text}\n\nReturn only 0, 1, 2 or 3."
    
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
response = [classify(prompt, x) for x in tqdm(text)]
print(response)

response = pd.Series(response, name="preds").astype(np.int64)
print(response)
response.value_counts()

all_synth = pd.concat([data, response], axis=1)
all_synth

all_synth.to_excel("GPT_synthetic_labeled_150_68.xlsx")



