
import re
import os
import pandas as pd
import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))  # path anchored for replication package


######
## 50
######

files = os.listdir("results/50_speeches")
files = [f for f in files if "speeches" in f]
files
len(files)

# F1_scores:
f1_scores = []
for i in range(0, len(files)):
    file = open("results/50_speeches/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
sd_roberta = pd.DataFrame(f1_scores).std()

results_roberta
sd_roberta

f1_scores_speeches = pd.concat([results_roberta.rename("ssmba_50")], axis=1)
# SEs for category 1 (CIs in the next figure file)
sds = pd.DataFrame(f1_scores).std()[1]
ses = sds / np.sqrt(10)

se_f1_p4 = pd.concat([pd.Series(ses).rename("ssmba_50")], axis=1)




######
## 75
######

files = os.listdir("results/75_speeches")
files = [f for f in files if "speeches" in f]
files
len(files)

# F1_scores:
f1_scores = []
for i in range(0, len(files)):
    file = open("results/75_speeches/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
sd_roberta = pd.DataFrame(f1_scores).std()

results_roberta
sd_roberta


f1_scores_speeches = pd.concat([f1_scores_speeches, results_roberta.rename("ssmba_75")], axis=1)
# SEs for category 1 (CIs in the next figure file)
sds = pd.DataFrame(f1_scores).std()[1]
ses = sds / np.sqrt(50)

se_f1_p4 = pd.concat([se_f1_p4, pd.Series(ses).rename("ssmba_75")], axis=1)


######
## 100
######

files = os.listdir("results/100_speeches")
files = [f for f in files if "speeches" in f]
files
len(files)

# F1_scores:
f1_scores = []
for i in range(0, len(files)):
    file = open("results/100_speeches/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
sd_roberta = pd.DataFrame(f1_scores).std()

results_roberta
sd_roberta

f1_scores_speeches = pd.concat([f1_scores_speeches, results_roberta.rename("ssmba_100")], axis=1)
# SEs for category 1 (CIs in the next figure file)
sds = pd.DataFrame(f1_scores).std()[1]
ses = sds / np.sqrt(50)

se_f1_p4 = pd.concat([se_f1_p4, pd.Series(ses).rename("ssmba_100")], axis=1)



######
## 150
######

files = os.listdir("results/150_speeches")
files = [f for f in files if "speeches" in f]
files
len(files)

# F1_scores:
f1_scores = []
for i in range(0, len(files)):
    file = open("results/150_speeches/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
sd_roberta = pd.DataFrame(f1_scores).std()

results_roberta
sd_roberta

f1_scores_speeches = pd.concat([f1_scores_speeches, results_roberta.rename("ssmba_150")], axis=1)
# SEs for category 1 (CIs in the next figure file)
sds = pd.DataFrame(f1_scores).std()[1]
ses = sds / np.sqrt(10)

se_f1_p4 = pd.concat([se_f1_p4, pd.Series(ses).rename("ssmba_150")], axis=1)


f1_scores_speeches
se_f1_p4

## EXPORT
f1_scores_speeches.to_excel("results/f1_scores_speeches_ssmba.xlsx")
se_f1_p4.to_excel("results/se_f1_speeches_ssmba.xlsx")






