
import pandas as pd
import numpy as np
import os
import pyreadr
import re

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # path anchored for replication package


### 50 / 150
data = pd.read_excel("data synthetic/synth_50_150.xlsx")

no_synth = data[data["synthetic"]==0]
no_synth.speechtype.value_counts()

no_synth.to_excel("data baselines/50_150_baseline.xlsx")


### 75 / 125
data = pd.read_excel("data synthetic/synth_75_125.xlsx")

no_synth = data[data["synthetic"]==0]
no_synth.speechtype.value_counts()

no_synth.to_excel("data baselines/75_125_baseline.xlsx")


### 100 / 100

data = pd.read_excel("data synthetic/synth_100_100.xlsx")

no_synth = data[data["synthetic"]==0]
no_synth.speechtype.value_counts()

no_synth.to_excel("data baselines/100_100_baseline.xlsx")


### 150 / 50

data = pd.read_excel("data synthetic/synth_150_50.xlsx")

no_synth = data[data["synthetic"]==0]
no_synth.speechtype.value_counts()
no_synth.synthetic.value_counts()

no_synth.to_excel("data baselines/150_50_baseline.xlsx")
















