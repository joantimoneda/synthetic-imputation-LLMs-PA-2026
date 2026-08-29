
import re
import os
import pandas as pd
import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))  # path anchored for replication package


######
## 50
######

files = os.listdir("results/50_101")
files = [f for f in files if "nostalgia" in f]
files
len(files)

# F1_scores:
f1_scores = []
for i in range(0, len(files)):
    file = open("results/50_101/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
sd_roberta = pd.DataFrame(f1_scores).std()

results_roberta
sd_roberta

f1_scores_nost = pd.concat([results_roberta.rename("ssmba_50")], axis=1)
# SEs for category 1 (CIs in the next figure file)
sds = pd.DataFrame(f1_scores).std()[1]
ses = sds / np.sqrt(len(files))

se_f1_nost = pd.concat([pd.Series(ses).rename("ssmba_50")], axis=1)



######
## 75
######

files = os.listdir("results/75_76")
files = [f for f in files if "nostalgia" in f]
files
len(files)

# F1_scores:
f1_scores = []
for i in range(0, len(files)):
    file = open("results/75_76/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
sd_roberta = pd.DataFrame(f1_scores).std()

results_roberta
sd_roberta


f1_scores_nost = pd.concat([f1_scores_nost, results_roberta.rename("ssmba_75")], axis=1)
# SEs for category 1 (CIs in the next figure file)
sds = pd.DataFrame(f1_scores).std()[1]
ses = sds / np.sqrt(len(files))

se_f1_nost = pd.concat([se_f1_nost, pd.Series(ses).rename("ssmba_75")], axis=1)



######
## Synth 75 76
######

files = os.listdir("results/100_51")
files = [f for f in files if "nostalgia" in f]
files
len(files)

# F1_scores:
f1_scores = []
for i in range(0, len(files)):
    file = open("results/100_51/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
sd_roberta = pd.DataFrame(f1_scores).std()

results_roberta
sd_roberta

f1_scores_nost = pd.concat([f1_scores_nost, results_roberta.rename("ssmba_100")], axis=1)
# SEs for category 1 (CIs in the next figure file)
sds = pd.DataFrame(f1_scores).std()[1]
ses = sds / np.sqrt(len(files))

se_f1_nost = pd.concat([se_f1_nost, pd.Series(ses).rename("ssmba_100")], axis=1)


f1_scores_nost
se_f1_nost

## EXPORT
f1_scores_nost.to_excel("results/f1_scores_nost_ssmba.xlsx")
se_f1_nost.to_excel("results/ses_f1_nost_ssmba.xlsx")






