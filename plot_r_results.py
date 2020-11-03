import pickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#manual inputting of data from excel sheet output from the R analysis
x_lch = pd.DataFrame(
    data=[
        ['random weights',0.1589,0.001,'Mantel test to LCH'],
        ['supervised',0.11167,0.001,'Mantel test to LCH'],
        ['{L,ab} (Tian et al.)',0.1195,0.001,'Mantel test to LCH'],
        ['Movie Lab',0.1403,0.001,'Mantel test to LCH'],
        ['SemanticCMC - 1 sec',0.03937,0.115,'Mantel test to LCH'],
        ['SemanticCMC - 10 sec',0.2121,0.001,'Mantel test to LCH'],
        ['SemanticCMC - 60 sec',0.2668,0.001,'Mantel test to LCH'],
        ['SemanticCMC - 5min',0.2302,0.001,'Mantel test to LCH'],
        ['random weights',1.89E-09,0.493,'Partial mantel to LCH \n (controlling for random)'],
        ['supervised',0.02595,0.206,'Partial mantel to LCH \n (controlling for random)'],
        ['{L,ab} (Tian et al.)',-0.02184,0.74,'Partial mantel to LCH \n (controlling for random)'],
        ['Movie Lab',0.0001595,0.484,'Partial mantel to LCH \n (controlling for random)'],
        ['SemanticCMC - 1 sec',-0.04642,0.931,'Partial mantel to LCH \n (controlling for random)'],
        ['SemanticCMC - 10 sec',0.1654,0.001,'Partial mantel to LCH \n (controlling for random)'],
        ['SemanticCMC - 60 sec',0.2195,0.001,'Partial mantel to LCH \n (controlling for random)'],
        ['SemanticCMC - 5min',0.1772,0.001,'Partial mantel to LCH \n (controlling for random)']
    ],
    columns = ['Network','Mantel Statistic (Pearson)','Significance','Controlled']
)

plt.close()
ax = sns.catplot(
    data=x_lch,
    kind='bar',
    x='Network',
    y='Mantel Statistic (Pearson)',
    hue='Controlled'
)

sns.set_style('whitegrid')
ax.despine(left=True, bottom=True)
ax.set_axis_labels("", "Mantel test statistic r (Pearson)")
ax.legend.set_title("")
ax.set_xticklabels(rotation=45, horizontalalignment='right')

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

plt.savefig('/home/clionaodoherty/Desktop/barplot.pdf',bbox_inches='tight')
plt.show()


