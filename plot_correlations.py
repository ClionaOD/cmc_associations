import seaborn as sns
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

num_classes = 256

with open(f'./results/{num_classes}_class_results.pickle','rb') as f:
    results = pickle.load(f)
results = {k:v for k,v in results.items() if not 'test' in k}

indexes = results[list(results.keys())[0]].index
cols = list(results.keys())

plot_df = pd.DataFrame(index=indexes, columns=cols)
for model, result_df in results.items():
    for idx, layer in enumerate(indexes):
        plot_df.loc[layer, model] = result_df.loc[layer, 'spearman correlation']

pval_df = pd.DataFrame(index=indexes, columns=cols)
for model, result_df in results.items():
    for idx, layer in enumerate(indexes):
        pval_df.loc[layer, model] = result_df.loc[layer, 'pval']

multiple_results = multipletests(
    pval_df.values.ravel(),
    alpha=0.01,
    method='bonferroni'
)

sig_df = pd.DataFrame(
    data= multiple_results[0].reshape(7,7),
    index=indexes, 
    columns=cols
)

fig, (ax1,leg) = plt.subplots(nrows=1,ncols=2,gridspec_kw={'width_ratios': [1,.3]}, figsize=(11.69/1.25,8.27/1.5))
fig.subplots_adjust(wspace=0.5)
sns.lineplot(data=plot_df.astype(float), ax=ax1, dashes=False)
handles, labels = ax1.get_legend_handles_labels()
ax1.get_legend().remove()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlabel('alexnet layer')
ax1.set_ylabel('correlation to lch matrix (spearman)')
leg.legend(handles, labels)
leg.axis('off') 

sigs=list(zip(np.where(sig_df==True)[0], np.where(sig_df==True)[1]))
for x in sigs:
    anot = (x[0], plot_df.iloc[x[0],x[1]]+.001)
    ax1.annotate('*', anot)

#plt.savefig('./results/lch_correlation_256_classes_withSig.pdf')
plt.show()
