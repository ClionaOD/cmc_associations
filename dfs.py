import pickle 
import pandas as pd

with open('./256_categs_lch.pickle','rb') as f:
    lch_df = pickle.load(f)
lch_vals = lch_df.values
lch_vals = -(lch_vals - lch_vals[0][0])
np.fill_diagonal(lch_vals,0)
lch_df = pd.DataFrame(data=lch_vals, index=lch_df.index, columns=lch_df.columns)
lch_df.to_csv('./results/lch_df.txt')

with open('./rdms/finetune60sec_rdms.pickle', 'rb') as f:
    rdms = pickle.load(f)
rdm = rdms['conv5']
rdm.to_csv('./results/60s_rdm_conv5_df.txt')

with open('./rdms/random_rdms.pickle', 'rb') as f:
    rand_rdms = pickle.load(f)
rand_rdm = rand_rdms['conv5']
rand_rdm.to_csv('./results/random_rdm_conv5_df.txt')

