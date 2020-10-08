import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from get_dend import hierarchical_clustering
from nltk.corpus import wordnet as wn

with open('./256_categs_lch.pickle','rb') as f:
    lch = pickle.load(f)

order = hierarchical_clustering(lch.values, lch.index)
order = [synset.split('.')[0] for synset in order]

for p in os.listdir('./rdms'):
    if not os.path.isdir(f'./rdms/{p}'):
        with open(f'./rdms/{p}','rb') as f:
            rdms = pickle.load(f)

        conv5 = rdms['conv5']

        wnids = list(conv5.index)
        synsets = []
        for wnid in wnids:
            pos = wnid[0]
            offset = int(wnid[1:])
            synsets.append(wn.synset_from_pos_and_offset(pos,offset).name())
        synsets = [synset.split('.')[0] for synset in synsets]

        conv5.index = synsets
        conv5.columns = synsets
        conv5 = conv5.reindex(index=order, columns=order)

        title = p.split('_')[0]

        fig, ax = plt.subplots(figsize=(12,12))
        sns.heatmap(conv5.astype(float), ax=ax)
        ax.tick_params('y',labelsize=7)
        ax.tick_params('x', labelsize=7)
        ax.set_title(title, fontsize=9)
        
        plt.savefig(f'./rdms/figs/{title}_rdm.pdf')
        #plt.show()
        plt.close()
