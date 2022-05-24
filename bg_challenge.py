import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from skbio.stats.distance import mantel
from sklearn.preprocessing import normalize
from scipy.spatial.distance import squareform
def normalise_matrix(M):
    M = (M - M.min()) / (M.max() - M.min())
    return M

def get_distrib_CI(corr_vals_1, corr_vals_2, alpha=0.05):
    d = corr_vals_1 - corr_vals_2
    d_sort=np.sort(d)
    d_len=len(d_sort)
    d_lower=d_sort[int(np.round(d_len * alpha/2))]
    d_upper=d_sort[int(np.round(d_len * (1-alpha/2)))]

    return [d_lower, d_upper]

rdm_folder = '/data/movie-associations/rdms/bg_challenge'
in9_types = ['original','mixed_rand','mixed_same','only_bg_t']

bootstrap_dir = '/data/movie-associations/bootstrapping'

layer = 'conv5'
training = 'main'

# lch_rdm = pd.read_csv('/data/movie-associations/rdms/semantic_models/imgnet_9_lch_distance.csv',index_col=0)
# # do min/max normalisation
# lch_rdm = normalise_matrix(lch_rdm.values)

#og_rdm.values - og_rdm.values.min() / (og_rdm.values.max() - og_rdm.values.min())

corr_df = pd.DataFrame(columns=['only background','mixed rand','mixed same','random vs same'])
p_df = pd.DataFrame(columns=['only background','mixed rand','mixed same','random vs same'])

for model in [i for i in os.listdir(os.path.join(rdm_folder,training,in9_types[0],layer)) if not 'random' in i]:
    
    corr_coefs = {}
    p_vals = {}

    print(f'...{model} correlations:')
    
    og_rdm = pd.read_csv(os.path.join(rdm_folder,training,in9_types[0],layer,model),index_col=0)
    # og_rdm = normalise_matrix(og_rdm.values)
    only_bg_rdm = pd.read_csv(os.path.join(rdm_folder,training,in9_types[3],layer,model),index_col=0)
    # only_bg_rdm = normalise_matrix(only_bg_rdm.values)
    mixed_rand_rdm = pd.read_csv(os.path.join(rdm_folder,training,in9_types[1],layer,model),index_col=0)
    # mixed_rand_rdm = normalise_matrix(mixed_rand_rdm.values)
    mixed_same_rdm = pd.read_csv(os.path.join(rdm_folder,training,in9_types[2],layer,model),index_col=0)
    # mixed_same_rdm = normalise_matrix(mixed_same_rdm.values)

    print('     original - only_bg test')
    og_onlybg_corr_coef, p_val, n = mantel(og_rdm, only_bg_rdm, method='pearson')
    print(f'        corr_coef: {og_onlybg_corr_coef}  p_val: {p_val}')
    corr_coefs['only background']=og_onlybg_corr_coef ; p_vals['only background']=p_val
    corr_distrib_og_onlybg = pd.read_pickle(f'mantel_bootstrap_results/rdm-level_{model.split("_")[0]}_original-vs-only_bg_t.pickle')
    print(f'        {corr_distrib_og_onlybg.mean()}')
    # sns.histplot(corr_distrib_og_onlybg)
    # plt.savefig(f'results/og-r-onlybg_{model.split("_")[0]}_distrib.png')
    # plt.close()

    print('     original - mixed_rand test')
    og_mr_corr_coef, p_val, n = mantel(og_rdm, mixed_rand_rdm, method='pearson')
    print(f'        corr_coef: {og_mr_corr_coef}  p_val: {p_val}')
    corr_coefs['mixed rand']=og_mr_corr_coef ; p_vals['mixed rand']=p_val
    corr_distrib_og_mixed_rand = pd.read_pickle(f'mantel_bootstrap_results/rdm-level_{model.split("_")[0]}_original-vs-mixed_rand.pickle')
    print(f'      {corr_distrib_og_mixed_rand.mean()}')
    sns.histplot(corr_distrib_og_mixed_rand)
    plt.savefig(f'results/og-r-mixedrand_{model.split("_")[0]}_distrib.png')
    plt.close()

    CI = get_distrib_CI(corr_distrib_og_mixed_rand,corr_distrib_og_onlybg)
    print(CI)
    if CI[0] < 0 < CI[1]:
        print('     n.s. diff')
    else:
        print('     sig at alphs=0.05')

    # print('     original - mixed_same test')
    # og_ms_corr_coef, p_val, n = mantel(og_rdm, mixed_same_rdm, method='pearson')
    # print(f'        corr_coef: {og_ms_corr_coef}  p_val: {p_val}')
    # corr_coefs['mixed same']=og_ms_corr_coef ; p_vals['mixed same']=p_val

    # print('     mixed_rand - mixed_same test')
    # mr_ms_corr_coef, p_val, n = mantel(mixed_rand_rdm, mixed_same_rdm, method='pearson')
    # print(f'        corr_coef: {mr_ms_corr_coef}  p_val: {p_val}')
    # corr_coefs['random vs same']=mr_ms_corr_coef ; p_vals['random vs same']=p_val
    # # bootstrap_results_pth = os.path.join(bootstrap_dir,f'{in9_types[0]}_{in9_types[2]}',f'{model.split("_")[0]}_exemplar-bootstrap.pickle')

    # corr_coefs['model']=model ; p_vals['model']=model
    # corr_df=corr_df.append(corr_coefs, ignore_index=True) ; p_df=p_df.append(p_vals, ignore_index=True)

    # # r(original, mixed_same) - r(original, mixed_rand)
    # print('     difference between mixed_same correlation and mixed_rand')
    # corr_diff = og_ms_corr_coef - og_mr_corr_coef
    # bootstrap_path = f'/data/movie-associations/bootstrapping/og_ms_less_og_mr/{model.split("_")[0]}_exemplar-bootstrap.pickle'
    # bootstrap_distrib = pd.read_pickle(bootstrap_path)
    # # CI = get_bootstrap_CI(bootstrap_distrib[layer])
    # # if corr_diff > CI[1]:
    # #     sig='yes'
    # # elif corr_diff < CI[0]:
    # #     sig='yes'
    # # else:
    # #     sig='no'
    # # print(f'         corr_diff: {corr_diff}  CI: {CI}  sig: {sig}')

    # # r(original, mixed_rand) - r(original, only_bg)
    # print('     difference between mixed_rand correlation and only background')
    # corr_diff = og_mr_corr_coef - og_onlybg_corr_coef
    # bootstrap_path = f'/data/movie-associations/bootstrapping/og_mr_less_ogonlybg/{model.split("_")[0]}_exemplar-bootstrap.pickle'
    # bootstrap_distrib = pd.read_pickle(bootstrap_path)

    # # sns.histplot(bootstrap_distrib['conv5'])
    # # plt.savefig(f'results/mr-minus-onlybg{model}_distrib.png')
    # # plt.close()
    # # CI = get_bootstrap_CI(bootstrap_distrib[layer])
    # # if corr_diff > CI[1]:
    # #     sig='yes'
    # # elif corr_diff < CI[0]:
    # #     sig='yes'
    # # else:
    # #     sig='no'
    # # print(f'         corr_diff: {corr_diff}  CI: {CI}  sig: {sig}')

    # # plot distributions for r(original, mixed_rand)
    # bootstrap_path = f'/data/movie-associations/bootstrapping/original_mixed_rand/{model.split("_")[0]}_exemplar-bootstrap.pickle'
    # bootstrap_distrib = pd.read_pickle(bootstrap_path)
    # sns.histplot(bootstrap_distrib['conv5'])
    # plt.savefig(f'results/og-r-mixedrand_{model}_distrib.png')
    # plt.close()

    # # plot distributions for r(original, only_bg)
    # bootstrap_path = f'/data/movie-associations/bootstrapping/original_only_bg/{model.split("_")[0]}_exemplar-bootstrap.pickle'
    # bootstrap_distrib = pd.read_pickle(bootstrap_path)
    # sns.histplot(bootstrap_distrib['conv5'])
    # plt.savefig(f'results/og-r-onlybg_{model}_distrib.png')
    # plt.close()

corr_df.to_csv('results/bg_challenge_pearson.csv',index=False)
p_df.to_csv('results/bg_challenge_pval.csv',index=False)