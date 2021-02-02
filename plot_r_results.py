import pickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mantel_df = pd.read_csv('./mantel_results.csv', sep=',')

#fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(ncols=2, nrows=2)

def plot(data, annotate=False):
    ax = sns.catplot(
        data=data,
        kind='bar',
        x='Network',
        y='Mantel Statistic (Pearson)',
        hue='Test Type',
        col='Image Type',
        col_wrap=2,
        legend_out=False
    )

    sns.set_style('whitegrid')
    ax.despine(left=True, bottom=True)
    ax.set_axis_labels("", "Mantel test statistic r (Pearson)")
    ax.set_xticklabels(rotation=45, horizontalalignment='right')
    ax.set(ylim=(-0.05,0.3)) #vals set from max and min of all results

    #this probably won't work wit current grid structure
    if annotate:
        for a in ax.axes.ravel():
            print(a)
            for p in a.patches:
                a.annotate(
                    '***', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', 
                    va = 'center', 
                    xytext = (0, 10), 
                    textcoords = 'offset points'
                )
    return ax

plot(mantel_df)
#plt.title('intact images')
plt.savefig('./figs/context_results.pdf',bbox_inches='tight')
plt.close()


