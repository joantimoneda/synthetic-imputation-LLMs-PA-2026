
import pandas as pd
import numpy as np
import os
import re

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))  # path anchored for replication package

data = pd.read_excel("nostalgia/nostalgia_data_clean_sample.xlsx", index_col=0) # use 
len(data)
data = data[["nostalgic", "text"]]
data.nostalgic.value_counts()
data = data.sample(frac=1).reset_index(drop=True)


# Step 1. Used translated English text
# Then take one category and: 
#   keep 50, then generate 150
#   keep 100, then generate 100
#   keep 150, then generate 50
#   could do it for 25, 75, 125, 175 as well, if an option


real = data[data["nostalgic"]==1]
real.reset_index(drop=True, inplace=True)
real

real.to_excel("nostalgia/data samples/real_nost_151.xlsx")

# sample of 50 speeches

np.random.seed(6)

real_sample = real.sample(n=50).reset_index(drop=True)
real_sample

real_sample.to_excel("nostalgia/data samples/real_nost_50.xlsx")

# sample of 75 speeches

np.random.seed(6)

real_sample_75 = real.sample(n=75).reset_index(drop=True)
real_sample_75

real_sample_75.to_excel("nostalgia/data samples/real_nost_75.xlsx")

# sample of 100 speeches

np.random.seed(6)

real_sample_100 = real.sample(n=100).reset_index(drop=True)
real_sample_100

real_sample_100.to_excel("nostalgia/data samples/real_nost_100.xlsx")





