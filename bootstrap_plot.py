from logging import root
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

root_dir = '/data/movie-associations/bootstrapping'

in9_type1 = 'original'
in9_type2 = 'mixed_rand'
in9_type3 = 'mixed_same'

#model = 'finetune1sec'
layers = 'conv5'

plot=False

def get_CI(corr_vals_1, corr_vals_2, alpha=0.05):
    #d = np.array(corr_vals_1) - np.array(corr_vals_2)
    d = np.array(corr_vals_1)
    d_sort=np.sort(d)
    d_len=len(d_sort)
    d_lower=d_sort[int(np.round(d_len * alpha/2))]
    d_upper=d_sort[int(np.round(d_len * (1-alpha/2)))]

    if d_lower<0<d_upper:
        return False
    else:
        return True

for model in os.listdir(os.path.join(root_dir,f'{in9_type1}_{in9_type3}')):
    print(model.split('_')[0])
    
    results_pth_1 = os.path.join(root_dir,f'{in9_type1}_{in9_type2}',model)
    results_pth_2 = os.path.join(root_dir,f'{in9_type1}_{in9_type3}',model)

    results_1 = pd.read_pickle(results_pth_1)
    corr_coefs_1 = [res[0] for res in results_1['conv5']]

    results_2 = pd.read_pickle(results_pth_2)
    corr_coefs_2 = [res[0] for res in results_2['conv5']]

    sig_dff = get_CI(corr_coefs_1,corr_coefs_2)
    if sig_dff:
        print('different')

    if plot:
        sns.displot(corr_coefs_1)
        plt.savefig(f'bootstrapping/{model.split("_")[0]}_{in9_type1}-vs-{in9_type2}.png')

        sns.displot(corr_coefs_2)
        plt.savefig(f'bootstrapping/{model.split("_")[0]}_{in9_type1}-vs-{in9_type3}.png')
