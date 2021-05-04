import pickle
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
import itertools

def get_coco_obj_synsets():
    with open('img_labels.pickle','rb') as f:
        labels = pickle.load(f)

    all = []
    img_order = []
    for img, lst in labels.items():
        for idx,word in enumerate(lst):
            if word == 'stop_sign':
                lst[idx] = 'street_sign'
            elif word == 'cell_phone':
                lst[idx] = 'cellular_telephone'

        syns = [wn.synsets(word) for word in lst]
        syns = [[j for j in i if j.pos()=='n'] for i in syns]
        all.append(syns)
        img_order.append(img)

    chosen_synsets = pd.read_csv('./wn_defs.csv')['syns'].to_list()
    chosen_synsets = [wn.synset(word) for word in chosen_synsets]

    img_synsets = [[[synset for synset in obj if synset in chosen_synsets] for obj in image] for image in all]
    img_synsets = [[item for sublst in image for item in sublst] for image in img_synsets]

    image_obj_synsets = list(zip(img_order,img_synsets))
    #with open('./MSCOCO_BOLD5000_object_synsets.pickle','wb') as f:
    #    pickle.dump(image_obj_synsets,f)
    
    return image_obj_synsets


def get_img_lch(img1,img2):
    combos = ((x,y) for x in img1 for y in img2)
    all_lch = []
    for pair in iter(combos):
        obj1 = pair[0]
        obj2 = pair[1]

        all_lch.append(obj1.lch_similarity(obj2))
    mean_lch = np.mean(all_lch)
    return mean_lch

def get_dataset_lch(img_order, img_obj_synsets):
    lch_df = pd.DataFrame(index=img_order,columns=img_order)
    empty = []
    for pair in itertools.product(img_obj_synsets,repeat=2):
        img1_file = pair[0][0]
        img1_labels = pair[0][1]
        
        img2_file = pair[1][0]
        img2_labels = pair[1][1]

        mean_lch = get_img_lch(set(img1_labels),set(img2_labels))
        
        lch_df.loc[img1_file][img2_file] = mean_lch
        lch_df.loc[img2_file][img1_file] = mean_lch
    
    return lch_df

if __name__ == '__main__':

    compute_synsets = True
    load_synsets = False

    if compute_synsets:
        image_obj_synsets = get_coco_obj_synsets()
    elif load_synsets:
        with open('./MSCOCO_BOLD5000_object_synsets.pickle','rb') as f:
            image_obj_synsets = pickle.load(f)

    img_order = [i[0] for i in image_obj_synsets]

    lch_df = get_dataset_lch(img_order, image_obj_synsets)

    lch_df.to_csv('/data/movie-associations/semantic_measures/MSCOCO_BOLD5000_2000_lch.csv')