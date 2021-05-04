import pickle
import pandas as pd
import os

rdm_pkl_path = 'rdms/segmentation/obj_trained/coco'
layers = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7']
save_path = '/data/movie-associations/rdms/obj_trained_coco'

models = [m for m in os.listdir(rdm_pkl_path) if '.pickle' in m]


for m in models:
    with open(f'{rdm_pkl_path}/{m}','rb') as f:
        model_rdms = pickle.load(f)
    
    _model = m.split('_')[0]
    for layer in layers:
        if not os.path.isdir(f'{save_path}/{layer}'):
            os.makedirs(f'{save_path}/{layer}')
        model_rdms[layer].to_csv(f'{save_path}/{layer}/{_model}_{layer}.csv')