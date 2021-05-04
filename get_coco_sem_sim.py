import numpy as np
import pickle
import pandas as pd
import itertools
from sentence_transformers import SentenceTransformer, util
from joblib import Parallel, delayed

model = SentenceTransformer('stsb-roberta-large')

model = [model]

with open('MSCOCO_BOLD5000_2000_annotations.pickle','rb') as f:
    coco_annots = pickle.load(f)

coco_annots = [(img,caption) for img,caption in coco_annots.items()]

img_order = [i[0] for i in coco_annots]
cosine_df = pd.DataFrame(index=img_order,columns=img_order)

def get_sem_sim(model, coco_annots, cosine_df):
    for pair in itertools.product(coco_annots,repeat=2):
        img1_file = pair[0][0]
        img1_annotation = pair[0][1]
        embedding1 = model.encode(img1_annotation, convert_to_tensor=True)
        
        img2_file = pair[1][0]
        img2_annotation= pair[1][1]
        embedding2 = model.encode(img2_annotation, convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)

        cosine_df.loc[img1_file][img2_file] = cosine_scores.cpu().numpy()[0][0]
        cosine_df.loc[img2_file][img1_file] = cosine_scores.cpu().numpy()[0][0]

    return cosine_df
    

sem_sim_df = Parallel(n_jobs=32)(delayed(get_sem_sim)(m, coco_annots, cosine_df) for m in model)

sem_sim_df[0].to_csv('/home/clionaodoherty/cmc_associations/MSCOCO_BOLD5000_2000_cosine_sims.csv')

