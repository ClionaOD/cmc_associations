import pickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import comp_corr
from skbio.stats.distance import mantel

with open('./results/256_class_results.pickle', 'rb') as f:
    results = pickle.load(f)
results = {k:v for k,v in results.items() if not 'test' in k}

conv5 = pd.DataFrame(index=list(results.keys()), columns=['spearman','pval','sig .01'])
for k,v in results.items():
    conv5.loc[k,'spearman'] = v.loc['conv5','spearman correlation']
    conv5.loc[k,'pval'] = v.loc['conv5','pval']

reord = ['random','authorLab','movieLab','finetune1sec','finetune10sec','finetune60sec']
conv5 = conv5.reindex(reord)

for n in reord:
    if conv5.loc[n,'pval'] < (0.01 / len(conv5)):
        conv5.loc[n]['sig .01'] = True

barplot = sns.barplot(x=conv5.index, y=conv5['spearman'])
for p, sig in zip(barplot.patches, conv5['sig .01']):
    if sig == True:
        barplot.text(p.get_x() + p.get_width() / 2., p.get_height(), '*', ha='center')

with open('./rdms/authorLab_rdms.pickle','rb') as f:
    lab = pickle.load(f)['conv5']
rdm = np.loadtxt('./results/60s_rdm_conv5_df.txt')
corr, pval, n = mantel(rdm, lab.values, method='spearman')
print(comp_corr.dependent_corr(corr,conv5.loc['finetune60sec','pval'],conv5.loc['authorLab','pval'], n=256*256))


