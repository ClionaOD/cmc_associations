import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

coefs = pd.read_csv('results/bg_challenge_pearson_condensed.csv')

sns.set(rc = {'figure.figsize':(20,20),})#'axes.facecolor':'cornflowerblue'

def plot(data, annotate=False):
    ax = sns.catplot(
        data=data,
        kind='bar',
        x='IN-9 Type',
        y='Pearson',
        col='Model',
        col_wrap=3,
        legend_out=False,
        hue='IN-9 Type',
        palette='magma'
    )

    sns.set_style('darkgrid')
    ax.despine(left=True, bottom=True)
    ax.set_axis_labels("", "Mantel test statistic r (Pearson)")
    ax.set_xticklabels([])
    #ax.set(ylim=(-0.1,1)) #vals set from max and min of all results

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

ax = plot(coefs)
#plt.gcf().set_size_inches(12, 12)
plt.savefig('results/test_violin.png',bbox_inches="tight")