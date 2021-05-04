import pickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

results_path = "/data/movie-associations/mantel_results/objtrain_imgnet_lch/mantel_imgnet_lch_conv5.csv"
n_models = 8

test_types = ['mantel' for i in range(n_models)]
test_types.extend(['partial mantel' for i in range(n_models)])

mantel_df = pd.read_csv(results_path)
mantel_df['Test Type'] = test_types

renames = {'authorLab_conv5.csv':'authorLab', 
    'finetune10sec-objtrain_conv5.csv':'finetune10sec-objtrain', 
    'finetune1sec-objtrain_conv5.csv':'finetune1sec-objtrain', 
    'finetune5min-objtrain_conv5.csv':'finetune5min-objtrain', 
    'finetune60sec-objtrain_conv5.csv':'finetune60sec-objtrain', 
    'movieLab_conv5.csv':'movieLab', 
    'random_conv5.csv':'random', 
    'supervised_conv5.csv':'supervised'
} 

mantel_df['Model']=[renames[model] for model in mantel_df['Model'].to_list()]

order = [
    renames['random_conv5.csv'],
    renames['supervised_conv5.csv'],
    renames['authorLab_conv5.csv'],
    renames['movieLab_conv5.csv'],
    renames['finetune1sec-objtrain_conv5.csv'],
    renames['finetune10sec-objtrain_conv5.csv'],
    renames['finetune60sec-objtrain_conv5.csv'],
    renames['finetune5min-objtrain_conv5.csv']
]

mantel_df = mantel_df.set_index('Model')
mantel_df = pd.concat([mantel_df[:n_models].reindex(order),mantel_df[n_models:].reindex(order)]).reset_index()

#fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(ncols=2, nrows=2)

def plot(data, annotate=False):
    ax = sns.catplot(
        data=data,
        kind='bar',
        x='Model',
        y='Pearson',
        hue='Test Type',
        legend_out=False
    )

    sns.set_style('whitegrid')
    ax.despine(left=True, bottom=True)
    ax.set_axis_labels("", "Mantel test statistic r (Pearson)")
    ax.set_xticklabels(rotation=45, horizontalalignment='right')
    ax.set(ylim=(-0.08,0.4)) #vals set from max and min of all results

    #this probably won't work wit current grid structure
    if annotate:
        for a in ax.axes.ravel():
            for idx, p in enumerate(a.patches):
                pval = data.loc[idx,'Sig']
                if pval <= 0.001:
                    an = '***'
                elif pval <= 0.01:
                    an = '**'
                elif pval <= 0.05:
                    an = '*'
                else:
                    an = ''
                
                a.annotate(
                    an, 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', 
                    va = 'center', 
                    xytext = (0, 10), 
                    textcoords = 'offset points'
                )
    return ax

plot(mantel_df, annotate=True)
#plt.title('intact images')
plt.savefig('./obj_trained_all.pdf',bbox_inches='tight')
plt.close()


