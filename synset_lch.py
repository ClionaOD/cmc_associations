import os
import json
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd

num_categories = 500
#category_path = f'/data/imagenet_cmc'
category_path = f'/home/clionaodoherty/Desktop/imagenet_categories'
list_path = f'{category_path}/imagenet_categs_{num_categories}.json'
test_wnids = []

if os.path.exists(list_path):
    with open(list_path, 'r') as f:
        test_wnids = json.load(f) 
else:
    print(f'no synset list found at {list_path}')
    raise FileNotFoundError

test_synsets = []
for wnid in test_wnids:
    pos = wnid[0]
    offset = int(wnid[1:])
    test_synsets.append(wn.synset_from_pos_and_offset(pos,offset))

lch_df = pd.DataFrame(columns=[_name.name() for _name in test_synsets], index=[_name.name() for _name in test_synsets])
for synset1 in test_synsets:
    for synset2 in test_synsets:
        lch_df.loc[synset1.name(), synset2.name()] = synset1.lch_similarity(synset2)

with open(f'{category_path}/lch_matrix_{num_categories}_categs.json','w') as f:
    json.dump(lch_df.to_json(orient='split'), f)

#TODO: map each synset in the list to a real label