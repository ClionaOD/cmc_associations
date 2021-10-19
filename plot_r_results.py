import pickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pth = '/data/movie-associations/mantel_results/main_imgnet_lch_blur/sigma10_kernel31'
test = 'mantel_imgnet_lch'
layers = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7']
#layers = ['conv5']
n_models = 10

obj_train = False

for layer in layers:
    results_path = f"{pth}/{test}_{layer}.csv"

    test_types = ['Correlation with LCH' for i in range(n_models)]
    test_types.extend(['Correlation with LCH (controlling random)' for i in range(n_models)])

    mantel_df = pd.read_csv(results_path)
    mantel_df['Test Type'] = test_types 

    if obj_train:
        renames = {f'authorLab':'{L,ab} - Tian et al.', 
                f'finetune10sec-objtrain':'SemanticCMC-objtrain - 10sec', 
                f'finetune1sec-objtrain':'SemanticCMC-objtrain - 1sec', 
                f'finetune5min-objtrain':'SemanticCMC-objtrain - 5min', 
                f'finetune60sec-objtrain':'SemanticCMC-objtrain - 60sec', 
                f'movieLab':'{L,ab} - movies', 
                f'random-distort':'random-distort',
                f'random-Lab':'random-Lab',
                f'random-supervised':'random-supervised', 
                f'supervised':'supervised'
            }
        order = [
            renames[f'random-distort'],
            renames[f'random-Lab'],
            renames[f'random-supervised'],
            renames[f'supervised'],
            renames[f'authorLab'],
            renames[f'movieLab'],
            renames[f'finetune1sec-objtrain'],
            renames[f'finetune10sec-objtrain'],
            renames[f'finetune60sec-objtrain'],
            renames[f'finetune5min-objtrain']
        ]

    else:
        renames = {f'authorLab':'{L,ab} - Tian et al.', 
                f'finetune10sec':'SemanticCMC - 10sec', 
                f'finetune1sec':'SemanticCMC - 1sec', 
                f'finetune5min':'SemanticCMC - 5min', 
                f'finetune60sec':'SemanticCMC - 60sec', 
                f'movieLab':'{L,ab} - movies', 
                f'random-distort':'random-distort',
                f'random-Lab':'random-Lab',
                f'random-supervised':'random-supervised', 
                f'supervised':'supervised'
            }
        order = [
            renames[f'random-distort'],
            renames[f'random-Lab'],
            renames[f'random-supervised'],
            renames[f'supervised'],
            renames[f'authorLab'],
            renames[f'movieLab'],
            renames[f'finetune1sec'],
            renames[f'finetune10sec'],
            renames[f'finetune60sec'],
            renames[f'finetune5min']
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
    fig_title = f'{test}_blur-sigma31_{layer}_barplot_correct_random.pdf'
    plt.savefig(f'./bar_figs/blur/{fig_title}',bbox_inches='tight')
    plt.close()