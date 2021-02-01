import pickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#manual inputting of data from excel sheet output from the R analysis

#published results, evaluaton on 256 ImageNet classes
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

#results when blurring using a gaussian filter with n_std=10.0
blur_lch = pd.DataFrame(
    data=[
        ['random weights',0.03477,0.184,'Mantel test to LCH'],
        ['supervised',None,None,'Mantel test to LCH'],
        ['{L,ab} (Tian et al.)',0.019666,0.32,'Mantel test to LCH'],
        ['Movie Lab',0.09133,0.007,'Mantel test to LCH'],
        ['SemanticCMC - 1 sec',-0.002675,0.518,'Mantel test to LCH'],
        ['SemanticCMC - 10 sec',0.2134,0.001,'Mantel test to LCH'],
        ['SemanticCMC - 60 sec',0.2136,0.001,'Mantel test to LCH'],
        ['SemanticCMC - 5min',0.2163,0.001,'Mantel test to LCH'],
        ['random weights',None,None,'Partial mantel to LCH \n (controlling for random)'],
        ['supervised',None,None,'Partial mantel to LCH \n (controlling for random)'],
        ['{L,ab} (Tian et al.)',0.01605,0.325,'Partial mantel to LCH \n (controlling for random)'],
        ['Movie Lab',0.1016,0.002,'Partial mantel to LCH \n (controlling for random)'],
        ['SemanticCMC - 1 sec',-0.03623,0.856,'Partial mantel to LCH \n (controlling for random)'],
        ['SemanticCMC - 10 sec',0.2172,0.001,'Partial mantel to LCH \n (controlling for random)'],
        ['SemanticCMC - 60 sec',0.2116,0.001,'Partial mantel to LCH \n (controlling for random)'],
        ['SemanticCMC - 5min',0.2155,0.001,'Partial mantel to LCH \n (controlling for random)']
    ],
    columns = ['Network','Mantel Statistic (Pearson)','Significance','Controlled']
)

#results when only objects are tested, segmented using DeepLab v3
segment_obj_lch = pd.DataFrame(
    data=[
        ['random weights',0.06292,0.009,'Mantel test to LCH'],
        ['supervised',None,None,'Mantel test to LCH'],
        ['{L,ab} (Tian et al.)',-9.28e-05,0.466,'Mantel test to LCH'],
        ['Movie Lab',0.003608,0.459,'Mantel test to LCH'],
        ['SemanticCMC - 1 sec',0.06673,0.023,'Mantel test to LCH'],
        ['SemanticCMC - 10 sec',0.02464,0.208,'Mantel test to LCH'],
        ['SemanticCMC - 60 sec',0.07329,0.011,'Mantel test to LCH'],
        ['SemanticCMC - 5min',0.04746,0.077,'Mantel test to LCH'],
        ['random weights',None,None,'Partial mantel to LCH \n (controlling for random)'],
        ['supervised',None,None,'Partial mantel to LCH \n (controlling for random)'],
        ['{L,ab} (Tian et al.)',-0.03681,0.822,'Partial mantel to LCH \n (controlling for random)'],
        ['Movie Lab',-0.02582,0.745,'Partial mantel to LCH \n (controlling for random)'],
        ['SemanticCMC - 1 sec',0.02626,0.228,'Partial mantel to LCH \n (controlling for random)'],
        ['SemanticCMC - 10 sec',-0.04529,0.937,'Partial mantel to LCH \n (controlling for random)'],
        ['SemanticCMC - 60 sec',0.04002,0.076,'Partial mantel to LCH \n (controlling for random)'],
        ['SemanticCMC - 5min',-0.01732,0.687,'Partial mantel to LCH \n (controlling for random)']
    ],
    columns = ['Network','Mantel Statistic (Pearson)','Significance','Controlled']
)

#results when only objects are tested, segmented using DeepLab v3
segment_bg_lch = pd.DataFrame(
    data=[
        ['random weights',0.06292,0.009,'Mantel test to LCH'],
        ['supervised',None,None,'Mantel test to LCH'],
        ['{L,ab} (Tian et al.)',-9.28e-05,0.466,'Mantel test to LCH'],
        ['Movie Lab',0.003608,0.459,'Mantel test to LCH'],
        ['SemanticCMC - 1 sec',0.06673,0.023,'Mantel test to LCH'],
        ['SemanticCMC - 10 sec',0.02464,0.208,'Mantel test to LCH'],
        ['SemanticCMC - 60 sec',0.07329,0.011,'Mantel test to LCH'],
        ['SemanticCMC - 5min',0.04746,0.077,'Mantel test to LCH'],
        ['random weights',None,None,'Partial mantel to LCH \n (controlling for random)'],
        ['supervised',None,None,'Partial mantel to LCH \n (controlling for random)'],
        ['{L,ab} (Tian et al.)',-0.03681,0.822,'Partial mantel to LCH \n (controlling for random)'],
        ['Movie Lab',-0.02582,0.745,'Partial mantel to LCH \n (controlling for random)'],
        ['SemanticCMC - 1 sec',0.02626,0.228,'Partial mantel to LCH \n (controlling for random)'],
        ['SemanticCMC - 10 sec',-0.04529,0.937,'Partial mantel to LCH \n (controlling for random)'],
        ['SemanticCMC - 60 sec',0.04002,0.076,'Partial mantel to LCH \n (controlling for random)'],
        ['SemanticCMC - 5min',-0.01732,0.687,'Partial mantel to LCH \n (controlling for random)']
    ],
    columns = ['Network','Mantel Statistic (Pearson)','Significance','Controlled']
)

#plt.close()

#fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(ncols=2, nrows=2)

def plot(data):
    ax = sns.catplot(
        data=data,
        kind='bar',
        x='Network',
        y='Mantel Statistic (Pearson)',
        hue='Controlled'
    )

    sns.set_style('whitegrid')
    ax.despine(left=True, bottom=True)
    ax.set_axis_labels("", "Mantel test statistic r (Pearson)")
    #ax.legend.set_title("")
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

plot(x_lch)
plt.title('intact images')
plt.savefig('./intact_barplot.pdf',bbox_inches='tight')
plt.close()

plot(blur_lch)
plt.title('blurred (sigma=10)')
plt.savefig('./blur_barplot.pdf',bbox_inches='tight')
plt.close()

plot(segment_obj_lch)
plt.title('segmented (objects only)')
plt.savefig('./segment_obj_barplot.pdf',bbox_inches='tight')
plt.close()

plot(segment_bg_lch)
plt.title('segmented (background only)')
plt.savefig('./segment_bg_barplot.pdf',bbox_inches='tight')
plt.close()

#plt.savefig('/home/clionaodoherty/Desktop/barplot.pdf',bbox_inches='tight')
plt.show()


