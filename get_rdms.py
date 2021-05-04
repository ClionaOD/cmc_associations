import pickle
import os
import json
import collections
import argparse

import pandas as pd
import numpy as np
import scipy.spatial.distance as ssd

def parse_option():
    parser = argparse.ArgumentParser('argument for activations')

    parser.add_argument('--image_path', type=str, help='path to images tested, for getting list of classes')
    parser.add_argument('--activation_path', type=str, help='path to directory of activations')
    parser.add_argument('--rdm_path', type=str, help='where to save rdms if save_rdm is True')
    parser.add_argument('--save_rdm', type=bool, default=True, help='whether to save the rdm dict')
    parser.add_argument('--single_layer', type=str, default=None, choices=['conv1','conv2','conv3','conv4','conv5','fc6','fc7'], help='layer for saving individual rdm')

    opt = parser.parse_args()

    return opt

def construct_activation_df(activations):
    """
    Returns the activations structured as a dict with {layer:pd.DataFrame(columns=classes)}
    activations: a dict of activations in response to numerous images. structured as {synset:{layers:}}
    """
    classes = list(activations.keys())
    layers = list(list(activations.values())[0].keys())
    activation_df = {k:pd.DataFrame(columns=classes) for k in layers}

    for k, v in activations.items():
        #k is the class, v is the dict with layers and activations
        for idx, l in enumerate(layers):
            _activationArray = v[l] #use indexing because blurred images saved first batch size index
            if idx <= 4:
                s = pd.Series(np.mean(_activationArray[0,:,:,:], axis=(1,2)))
                activation_df[l][k] = s
            else:
                s = pd.Series(_activationArray[0,:])
                activation_df[l][k] = s
    
    return activation_df

def construct_rdm(activation_df):
    """
    Takes a df with columns=classes, series=mean activations and returns the data as an n*n rdm dataframe
    activation_df: a df with columns=classes)
    """
    rdm = ssd.pdist(activation_df.values.T, metric='cosine')
    rdm = ssd.squareform(rdm)
    rdm = pd.DataFrame(rdm, columns=activation_df.columns, index=activation_df.columns)
    return rdm

def main(args):
    for a in os.listdir(args.activation_path):
        if not os.path.isdir(f'{args.activation_path}/{a}'):
            with open(f'{args.activation_path}/{a}', 'rb') as f:
                acts = pickle.load(f)
            
            activation_dfs = construct_activation_df(acts)
            rdm_dict = {k:construct_rdm(v) for k,v in activation_dfs.items()}

            if args.save_rdm:
                _act_type = args.activation_path.split('/')[-1]
                _rdm_save_path = f'{args.rdm_path}/{_act_type}'
                if not os.path.isdir(_rdm_save_path):
                    os.makedirs(_rdm_save_path)
                _save = a.split('_')[0]
                with open(f'{_rdm_save_path}/{_save}_rdms.pickle','wb') as f:
                    pickle.dump(rdm_dict,f)
                
                if type(args.single_layer) == list:
                    for layer in args.single_layer:
                        if not os.path.isdir(f'{_rdm_save_path}/{layer}/'):
                            os.makedirs(f'{_rdm_save_path}/{layer}/')
                        rdm_dict[layer].to_csv(f'{_rdm_save_path}/{layer}/{_save}_{layer}.csv')
                elif type(args.single_layer) == str:
                    layer = args.single_layer
                    if not os.path.isdir(f'{_rdm_save_path}/{layer}/'):
                        os.makedirs(f'{_rdm_save_path}/{layer}/')
                    rdm_dict[layer].to_csv(f'{_rdm_save_path}/{layer}/{_save}_{layer}.csv')
                else:
                    pass

if __name__ == "__main__":
    args = parse_option()
    args.image_path = '/data/movie-associations/MSCOCO_BOLD5000_2000'
    args.activation_path = '/data/movie-associations/activations/segmentation/obj_trained/coco/rep_3_all/'  
    args.rdm_path = '/data/movie-associations/rdms/segmentation/obj_trained/coco'
    args.single_layer = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7']
    #args.single_layer = 'conv5'
    main(args)
