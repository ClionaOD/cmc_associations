import pandas as pd
import os
from skbio.stats.distance import mantel
from sklearn.preprocessing import normalize
from scipy.spatial.distance import squareform

rdm_folder = '/data/movie-associations/rdms/bg_challenge'
in9_types = ['original','mixed_rand','mixed_same','only_bg_t']

layer = 'conv5'

lch_rdm = pd.read_csv('/data/movie-associations/semantic_measures/imgnet_9_lch.csv',index_col=0)

#og_rdm.values - og_rdm.values.min() / (og_rdm.values.max() - og_rdm.values.min())

for model in [i for i in os.listdir(os.path.join(rdm_folder,in9_types[0],layer))]:
    
    print(f'...{model} correlations:')
    
    og_rdm = pd.read_csv(os.path.join(rdm_folder,in9_types[0],layer,model),index_col=0)
    only_bg_rdm = pd.read_csv(os.path.join(rdm_folder,in9_types[3],layer,model),index_col=0)
    mixed_rand_rdm = pd.read_csv(os.path.join(rdm_folder,in9_types[1],layer,model),index_col=0)
    mixed_same_rdm = pd.read_csv(os.path.join(rdm_folder,in9_types[2],layer,model),index_col=0)

    # results_df = pd.DataFrame(columns=['original vs. only bg','original vs. random background','original vs. same background'])

    print('     original - only_bg test')
    corr_coef, p_val, n = mantel(og_rdm, only_bg_rdm, method='kendalltau')
    print(f'        corr_coef: {corr_coef}  p_val: {p_val}')

    print('     original - mixed_rand test')
    corr_coef, p_val, n = mantel(og_rdm, mixed_rand_rdm, method='kendalltau')
    print(f'        corr_coef: {corr_coef}  p_val: {p_val}')

    # print('     original - mixed_same test')
    # corr_coef, p_val, n = mantel(og_rdm, mixed_same_rdm, method='kendalltau')
    # print(f'        corr_coef: {corr_coef}  p_val: {p_val}')

    print('     mixed_rand - mixed_same test')
    corr_coef, p_val, n = mantel(mixed_rand_rdm, mixed_same_rdm, method='kendalltau')
    print(f'        corr_coef: {corr_coef}  p_val: {p_val}')

    # print('     original vs. LCH')
    # corr_coef, p_val, n = mantel(og_rdm, lch_rdm, method='kendalltau')
    # print(f'        corr_coef: {corr_coef}  p_val: {p_val}')

    # print('     only_bg vs. LCH')
    # corr_coef, p_val, n = mantel(only_bg_rdm, lch_rdm, method='kendalltau')
    # print(f'        corr_coef: {corr_coef}  p_val: {p_val}')

    # print('     mixed_rand vs. LCH')
    # corr_coef, p_val, n = mantel(mixed_rand_rdm, lch_rdm, method='kendalltau')
    # print(f'        corr_coef: {corr_coef}  p_val: {p_val}')

    # print('     mixed_same vs. LCH')
    # corr_coef, p_val, n = mantel(mixed_same_rdm, lch_rdm, method='kendalltau')
    # print(f'        corr_coef: {corr_coef}  p_val: {p_val}')