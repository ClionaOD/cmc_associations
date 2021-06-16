import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

partial = True
n_models = 8
p = 0.05 

mantel_path = '/data/movie-associations/mantel_results/objtrain_imgnet_brain'
layers = ['conv1','conv2', 'conv3','conv4','conv5','fc6','fc7']
rois = ['RHPPA', 'LHEarlyVis', 'LHOPA', 'RHEarlyVis', 'LHPPA', 'LHLOC', 'RHRSC', 'RHOPA', 'RHLOC', 'LHRSC']
models = ['random', 'supervised',  'authorLab', 'movieLab', 'finetune1sec-objtrain', 'finetune10sec-objtrain', 'finetune60sec-objtrain', 'finetune5min-objtrain']

fig, ((LH_EarlyVis,RH_EarlyVis),(LH_LOC,RH_LOC),(LH_OPA,RH_OPA),(LH_PPA,RH_PPA),(LH_RSC,RH_RSC)) = plt.subplots(nrows=5,ncols=2,figsize=(8.27,11.69), sharex=True, sharey=True)

results_per_layer = {layer:None for layer in layers}
for layer in layers:
    results_per_roi = {roi_file.split('_')[0]:pd.read_csv(f'{mantel_path}/{layer}/{roi_file}') for roi_file in os.listdir(f'{mantel_path}/{layer}')}
    results_per_layer[layer] = results_per_roi

#include full mantel only
if partial:
    results_per_layer = {layer:{roi:df[n_models:] for roi,df in roi_dict.items()} for layer,roi_dict in results_per_layer.items()}
else:
    results_per_layer = {layer:{roi:df[:n_models] for roi,df in roi_dict.items()} for layer,roi_dict in results_per_layer.items()}

for idx, roi in enumerate(rois):
    roi_stat_df = pd.DataFrame(index=models, columns=layers)
    roi_sig_df = pd.DataFrame(index=models, columns=layers)
    for layer in layers:
        for model in models:
            df = results_per_layer[layer][roi]
            stat = df.loc[df['Model'] == model]['Pearson']
            sig = df.loc[df['Model'] == model]['Sig']

            roi_stat_df.loc[model,layer] = stat.values[0]
            if sig.values[0] < p:
                roi_sig_df.loc[model,layer] = 1
            else:
                roi_sig_df.loc[model,layer] = 0
    
    if roi == 'LHEarlyVis':
        ax = LH_EarlyVis
    elif roi == 'RHEarlyVis':
        ax = RH_EarlyVis
    elif roi == 'LHLOC':
        ax = LH_LOC
    elif roi == 'RHLOC':
        ax = RH_LOC
    elif roi == 'LHOPA':
        ax = LH_OPA
    elif roi == 'RHOPA':
        ax = RH_OPA
    elif roi == 'LHPPA':
        ax = LH_PPA
    elif roi == 'RHPPA':
        ax = RH_PPA
    elif roi == 'LHRSC':
        ax = LH_RSC
    elif roi == 'RHRSC':
        ax = RH_RSC

    sns.lineplot(data=roi_stat_df.T.astype(float), ax=ax, dashes=False)
    if not ax == LH_EarlyVis:
        ax.get_legend().remove()
    ax.set_title(roi)
    ax.set_ylim([-0.03,0.04])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sigs=list(zip(np.where(roi_sig_df==1)[0], np.where(roi_sig_df==1)[1]))
    for x in sigs:
        anot = (x[1] , roi_stat_df.iloc[x[0]][x[1]])
        ax.annotate('*', anot)

title = f'objtrain_imgnet_brain_p_{p}_uncorrected.jpg'
if partial:
    title = 'partial_'+title
plt.savefig(f'./{title}')