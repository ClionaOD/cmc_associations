import os
import pandas as pd
import numpy as np
import pickle 

from get_rdms import construct_activation_df, construct_rdm

from scipy.stats import pearsonr
from scipy.spatial.distance import squareform

training = 'main'

tests = [['original','only_bg_t'],['original','mixed_rand']]

for test in tests:
    activation_path_1 = f'/data/movie-associations/activations/bg_challenge/{training}/{test[0]}'
    activation_path_2 = f'/data/movie-associations/activations/bg_challenge/{training}/{test[1]}'

    for a in [i for i in os.listdir(activation_path_1) if '.pickle' in i]:
        acts_1=pd.read_pickle(os.path.join(activation_path_1,a))
        acts_2=pd.read_pickle(os.path.join(activation_path_2,a))

        activation_dfs_1 = construct_activation_df(acts_1)
        activation_dfs_2 = construct_activation_df(acts_2)

        rdm_1 = construct_rdm(activation_dfs_1['conv5'])
        rdm_2 = construct_rdm(activation_dfs_2['conv5'])

        corr_coefs = []
        for i in range(1000):
            bootstrap_choices = np.random.choice(9,9,replace=True)
            
            rdm_1_boot = rdm_1.values[bootstrap_choices,:][:,bootstrap_choices]
            rdm_2_boot = rdm_2.values[bootstrap_choices,:][:,bootstrap_choices]

            corr_coef, p_val = pearsonr(squareform(rdm_1_boot), squareform(rdm_2_boot))
            corr_coefs.append(corr_coef)
        
        corr_coefs=np.array(corr_coefs)
        with open(f'mantel_bootstrap_results/rdm-level_{a.split("_")[0]}_{test[0]}-vs-{test[1]}.pickle','wb') as f:
            pickle.dump(corr_coefs,f)

