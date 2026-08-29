
import pandas as pd
import numpy as np

d1 = pd.read_excel("../data synthetic/synthetic_50_101.xlsx", index_col=0)
d1

d2 = pd.read_excel("../../data samples/real_nost_50.xlsx", index_col=0)
d2

# length synthetic: 

lengths = [len(x.split(" ")) for x in d1.text]
lengths
np.max(lengths)
np.mean(lengths)
# 20.39 words

lengths_real = [len(x.split(" ")) for x in d2.text]
lengths_real
np.max(lengths_real)
np.mean(lengths_real)
# 21.3 words



## First sentence only:
sent = [x.split(".")[0] for x in d1.text]
lengths = [len(x.split(" ")) for x in sent]

lengths = [len(x.split(" ")) for x in d1.text]
lengths
np.max(lengths)
np.mean(lengths)
# 43 words

