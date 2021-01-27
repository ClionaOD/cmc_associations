import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from nltk.corpus import wordnet as wn
import pandas as pd
from scipy.spatial import procrustes
from sklearn.manifold import MDS
import numpy as np

with open(f'./rdms/authorLab_rdms.pickle','rb') as f:
    rdms = pickle.load(f)

conv5 = rdms['conv5']

wnids = list(conv5.index)
synsets = []
for wnid in wnids:
    pos = wnid[0]
    offset = int(wnid[1:])
    synsets.append(wn.synset_from_pos_and_offset(pos,offset))

hypernyms = {synset: synset.hypernyms() for synset in synsets}
for k,v in hypernyms.items():
    hypernyms[k] = v[0]

groups = {v:[] for v in hypernyms.values()}
for k,v in hypernyms.items():
    groups[v].append(k)

plot_groups = {k.name().split('.')[0] : [s.name().split('.')[0] for s in v] for k,v in groups.items() if len(v) > 1}
groups_df = pd.DataFrame.from_dict(plot_groups, orient='index')

x = []
for i in groups_df.values:
    for j in i:
       if not j is None:
           x.append(j) 

ref = False

for p in os.listdir('./rdms'):
    if not os.path.isdir(f'./rdms/{p}'):
        title = p.split('_')[0]

        with open(f'./rdms/{p}','rb') as f:
            rdms = pickle.load(f)
        
        rdm = rdms['conv5']
        rdm.index = [synset.name().split('.')[0] for synset in synsets]
        rdm.columns = [synset.name().split('.')[0] for synset in synsets]

        #do MDS on the euclidean distance matrix, set category for plot 
        mds_results = {}

        mds = MDS(n_components=2, dissimilarity='precomputed')
        df_embedding = pd.DataFrame(mds.fit_transform(rdm.values), index=rdm.index)
        if ref == False:
            align = df_embedding.values
            ref = True
        else:
            mtx1, mtx2, disparity = procrustes(align, df_embedding.values)
            df_embedding = pd.DataFrame(mtx2, index=rdm.index)
        mds_results['full embedding'] = df_embedding

        hypernym_df_embedding = pd.DataFrame(index=x, columns=[0,1,'hypernym'])
        for i in hypernym_df_embedding.index:
            hypernym_df_embedding.loc[i,0] = df_embedding.loc[i,0]
            hypernym_df_embedding.loc[i,1] = df_embedding.loc[i,1]
            hypernym_df_embedding.loc[i,'hypernym'] = list(groups_df.index)[np.where(groups_df == i)[0][0]]
        mds_results['hypernym embedding'] = hypernym_df_embedding

        with open(f'./mds/{title}_mds.pickle','wb') as f:
            pickle.dump(mds_results,f)
        
        fig, ax = plt.subplots(figsize=(12,12))
        sns.scatterplot(
            x=hypernym_df_embedding[0],
            y=hypernym_df_embedding[1],
            hue=hypernym_df_embedding['hypernym'], 
            legend=True,  
            ax=ax)
        ax.set_title(title)
        ax.set_xlabel(' ')
        ax.set_ylabel(' ')
        ax.axis('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #plt.show()
        plt.savefig(f'./mds/{title}_mds.pdf')
        plt.close()
                