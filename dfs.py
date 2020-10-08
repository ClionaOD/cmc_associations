import pickle 
import pandas as pd
import numpy as np

with open('./256_categs_lch.pickle','rb') as f:
    lch_df = pickle.load(f)
lch_vals = lch_df.values
lch_vals = -(lch_vals - lch_vals[0][0])
np.fill_diagonal(lch_vals,0)
np.savetxt('./results/lch_df.txt', lch_vals)

with open('./rdms/finetune60sec_rdms.pickle', 'rb') as f:
    rdms = pickle.load(f)
rdm = rdms['conv5'].values
np.savetxt('./results/60s_rdm_conv5_df.txt', rdm)

with open('./rdms/random_rdms.pickle', 'rb') as f:
    rand_rdms = pickle.load(f)
rand_rdm = rand_rdms['conv5'].values
np.savetxt('./results/random_rdm_conv5_df.txt',rand_rdm)

