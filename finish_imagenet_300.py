import os
import pickle
import json
from get_imagenet import url_to_image, download_picture, iter_synsets
from joblib import Parallel, delayed

final_synsets={'n0781842' : 145,
    'n10639637' : 140,
    'n01687128': 142,
    'n07815294': 142,
    'n07583865': 138, 
    'n04396335': 133, 
    'n01534762': 130, 
    'n01981702': 137, 
    'n03542727': 134, 
    'n03246197': 133, 
    'n04562122': 139
}

num_categories = 300
num_images = 150
category_path = f'/data/imagenet_cmc'

test_categs = [k for k in final_synsets.keys()]

Parallel(n_jobs=32)(delayed(iter_synsets)(synset, category_path) for synset in test_categs)
