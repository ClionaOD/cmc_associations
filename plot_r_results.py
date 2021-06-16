import pickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pth = '/data/movie-associations/mantel_results/main_imgnet_lch/rep_2'
test = 'mantel_imgnet_lch_COMPILED'
#layers = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7']
layers = ['conv5']
n_models = 8

obj_train = False

for layer in layers:
    results_path = f"{pth}/{test}_{layer}.csv"

    test_types = ['Correlation with LCH' for i in range(n_models)]
    test_types.extend(['Correlation with LCH (controlling random)' for i in range(n_models)])

    mantel_df = pd.read_csv(results_path)
    mantel_df['Test Type'] = test_types 

    if obj_train:
        renames = {f'authorLab_{layer}.csv':'authorLab', 
                f'finetune10sec-objtrain_{layer}.csv':'finetune10sec-objtrain', 
                f'finetune1sec-objtrain_{layer}.csv':'finetune1sec-objtrain', 
                f'finetune5min-objtrain_{layer}.csv':'finetune5min-objtrain', 
                f'finetune60sec-objtrain_{layer}.csv':'finetune60sec-objtrain', 
                f'movieLab_{layer}.csv':'movieLab', 
                f'random_{layer}.csv':'random', 
                f'supervised_{layer}.csv':'supervised'
            }
        order = [
            renames[f'random_{layer}.csv'],
            renames[f'supervised_{layer}.csv'],
            renames[f'authorLab_{layer}.csv'],
            renames[f'movieLab_{layer}.csv'],
            renames[f'finetune1sec-objtrain_{layer}.csv'],
            renames[f'finetune10sec-objtrain_{layer}.csv'],
            renames[f'finetune60sec-objtrain_{layer}.csv'],
            renames[f'finetune5min-objtrain_{layer}.csv']
        ]

    else:
        renames = {f'authorLab_{layer}.csv':'{L,ab} - Tian et al.', 
                f'finetune10sec_{layer}.csv':'SemanticCMC - 10sec', 
                f'finetune1sec_{layer}.csv':'SemanticCMC - 1sec', 
                f'finetune5min_{layer}.csv':'SemanticCMC - 5min', 
                f'finetune60sec_{layer}.csv':'SemanticCMC - 60sec', 
                f'movieLab_{layer}.csv':'{L,ab} - movies', 
                f'random_{layer}.csv':'random', 
                f'supervised_{layer}.csv':'supervised'
            }
        order = [
            renames[f'random_{layer}.csv'],
            renames[f'supervised_{layer}.csv'],
            renames[f'authorLab_{layer}.csv'],
            renames[f'movieLab_{layer}.csv'],
            renames[f'finetune1sec_{layer}.csv'],
            renames[f'finetune10sec_{layer}.csv'],
            renames[f'finetune60sec_{layer}.csv'],
            renames[f'finetune5min_{layer}.csv']
        ]

    mantel_df['Model']=[renames[model] for model in mantel_df['Model'].to_list()]
    mantel_df = mantel_df.set_index('Model')
    mantel_df = pd.concat([mantel_df[:n_models].reindex(order),mantel_df[n_models:].reindex(order)]).reset_index()

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
    fig_title = f'{test}_{layer}_barplot_publish_rep2.pdf'
    plt.savefig(f'./bar_figs/{fig_title}',bbox_inches='tight')
    plt.close()