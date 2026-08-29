
import pandas as pd
import numpy as np
import os
import pyreadr
import re

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # path anchored for replication package


### 50 / 150
data = pd.read_excel("data synthetic/synthetic_50_101_fullsample.xlsx", index_col=0)

no_synth = data[data["synthetic"]==0]
no_synth.nostalgic.value_counts()

no_synth.to_excel("data baselines/50_baseline.xlsx")


### 75 / 125
data = pd.read_excel("data synthetic/synthetic_75_76_fullsample.xlsx")

no_synth = data[data["synthetic"]==0]
no_synth.nostalgic.value_counts()

no_synth.to_excel("data baselines/75_baseline.xlsx")


### 100 / 100

data = pd.read_excel("data synthetic/synthetic_100_51_fullsample.xlsx")

no_synth = data[data["synthetic"]==0]
no_synth.nostalgic.value_counts()

no_synth.to_excel("data baselines/100_baseline.xlsx")
















