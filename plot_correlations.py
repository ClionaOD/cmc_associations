import seaborn as sns
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

num_classes = 256

with open(f'./results/{num_classes}_class_results_pearson.pickle','rb') as f:
    results = pickle.load(f)
results = {k:v for k,v in results.items() if not 'test' in k and not 'figs' in k}

indexes = results[list(results.keys())[0]].index
cols = list(results.keys())

plot_df = pd.DataFrame(index=indexes, columns=cols)
for model, result_df in results.items():
    for idx, layer in enumerate(indexes):
        plot_df.loc[layer, model] = result_df.loc[layer, 'pearson correlation']

re_col = ['SemanticCMC-1sec','SemanticCMC-60sec','supervised','SemanticCMC-5min','Movie {L,ab}', '{L,ab} (Tian et al.)','SemanticCMC-10sec','random weights']

pval_df = pd.DataFrame(index=indexes, columns=cols)
for model, result_df in results.items():
    for idx, layer in enumerate(indexes):
        pval_df.loc[layer, model] = result_df.loc[layer, 'pval']

multiple_results = multipletests(
    pval_df.values.ravel(),
    alpha=0.05,
    method='bonferroni'
)

sig_df = pd.DataFrame(
    data= multiple_results[0].reshape(7,8),
    index=indexes, 
    columns=cols
)

re_col = ['SemanticCMC-1sec','SemanticCMC-60sec','supervised','SemanticCMC-5min','Movie {L,ab}', '{L,ab} (Tian et al.)','SemanticCMC-10sec','random weights']
plot_df.columns=re_col
sig_df.columns=re_col

reord = ['random weights','supervised', '{L,ab} (Tian et al.)','Movie {L,ab}','SemanticCMC-1sec','SemanticCMC-10sec','SemanticCMC-60sec','SemanticCMC-5min']
plot_df = plot_df.reindex(columns=reord)
sig_df = sig_df.reindex(columns=reord)

fig, (ax1,leg) = plt.subplots(nrows=1,ncols=2,gridspec_kw={'width_ratios': [1,.3]}, figsize=(11.69/1.25,8.27/1.5))
fig.subplots_adjust(wspace=0.5)
sns.lineplot(data=plot_df.astype(float), ax=ax1, dashes=False)
handles, labels = ax1.get_legend_handles_labels()
ax1.get_legend().remove()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlabel('alexnet layer')
ax1.set_ylabel('coding of LCH similarity')
leg.legend(handles, labels)
leg.axis('off') 

#sigs=list(zip(np.where(sig_df==True)[0], np.where(sig_df==True)[1]))
sigs=list(zip(np.where(pval_df<(.05/7))[0], np.where(pval_df<(.05/7))[1]))
for x in sigs:
    anot = (x[0], plot_df.iloc[x[0],x[1]]+.001)
    ax1.annotate('*', anot)

plt.savefig('./results/pearson_lch_correlation.pdf')
plt.show()
