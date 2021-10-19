import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from nltk.corpus import wordnet as wn
import pandas as pd
from scipy.spatial import procrustes
from sklearn.manifold import MDS
import numpy as np

rdm_pth = '/data/movie-associations/rdms/main/correct_random'

models = [i for i in os.listdir(rdm_pth) if 'pickle' in i]

for model in models:
    with open(f'{rdm_pth}/{model}','rb') as f:
        rdms = pickle.load(f)

    conv5 = rdms['conv5']

    wnids = list(conv5.index)
    synsets = []
    for wnid in wnids:
        pos = wnid[0]
        offset = int(wnid[1:])
        synsets.append(wn.synset_from_pos_and_offset(pos,offset))

    # code for getting hypernyms and arranging according to this
    # nice idea but most have a unique hypernym so it doesn't work in practice
    """hypernyms = {synset: synset.hypernyms() for synset in synsets}
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
                x.append(j) """
    
    # new option, manually assigned superordinate categories for initial exploration
    superords = pd.read_csv('synsets_manual_categs.csv')
    chosen_syns = [s.split('\'')[1].split('.')[0] for s in superords.SYNSET.to_list()]
    chosen_syns[chosen_syns.index('planter')] = 'planter\'s_punch'
    chosen_superords = superords.MINE.to_list()
    groups = dict(zip(chosen_syns,chosen_superords))

    ref = False

    for model in [r for r in os.listdir(rdm_pth) if '.pickle' in r]:
        model_name = model.split('_')[0]

        rdm = pd.read_pickle(f'{rdm_pth}/{model}')
        
        rdm = rdms['conv5']
        rdm.index = [synset.name().split('.')[0] for synset in synsets]
        rdm.columns = [synset.name().split('.')[0] for synset in synsets]

        #subselect to only those manually selected (180 x 180)
        rdm = rdm.reindex(index=chosen_syns,columns=chosen_syns)

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
        
        df_embedding['category'] = chosen_superords
        mds_results['full embedding'] = df_embedding

        # legacy option for hypernym method
        """hypernym_df_embedding = pd.DataFrame(index=x, columns=[0,1,'hypernym'])
        for i in hypernym_df_embedding.index:
            hypernym_df_embedding.loc[i,0] = df_embedding.loc[i,0]
            hypernym_df_embedding.loc[i,1] = df_embedding.loc[i,1]
            hypernym_df_embedding.loc[i,'hypernym'] = list(groups_df.index)[np.where(groups_df == i)[0][0]]
        mds_results['hypernym embedding'] = hypernym_df_embedding"""

        with open(f'./mds/manual_superords/{model_name}_conv5_mds.pickle','wb') as f:
            pickle.dump(mds_results,f)
        
        fig, ax = plt.subplots(figsize=(12,12))
        sns.scatterplot(
            x=df_embedding[0],
            y=df_embedding[1],
            hue=df_embedding['category'], 
            legend=True,  
            ax=ax)
        ax.set_title(model_name)
        ax.set_xlabel(' ')
        ax.set_ylabel(' ')
        ax.axis('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #plt.show()
        plt.savefig(f'./mds/manual_superords/{model_name}_mds.pdf')
        plt.close()
                