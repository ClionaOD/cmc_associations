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
    parser.add_argument('--save_rdm', type=bool, default=False, help='whether to save the rdm dict')

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
            _activationArray = v[l]
            if not idx > 4:
                s = pd.Series(np.mean(_activationArray, axis=(1,2)))
                activation_df[l][k] = s
            else:
                s = pd.Series(_activationArray)
                activation_df[l][k] = s
    
    return activation_df

def construct_rdm(activation_df):
    """
    Takes a df with columns=classes, series=mean activations and returns the data as an n*n rdm dataframe
    activation_df: a df with columns=classes)
    """
    rdm = ssd.pdist(activation_df.values.T)
    rdm = ssd.squareform(rdm)
    rdm = pd.DataFrame(rdm, columns=activation_df.columns, index=activation_df.columns)
    return rdm

def get_rdms(args):
    for a in os.listdir(args.activation_path):
        with open(f'{args.activation_path}/{a}', 'rb') as f:
            acts = pickle.load(f)
        
        activation_dfs = construct_activation_df(acts)
        rdm_dict = {k:construct_rdm(v) for k,v in activation_dfs.items()}

        if args.save_rdm:
            _save = a.split('_')[0]
            with open(f'{args.rdm_path}/{_save}_rdms.pickle','wb') as f:
                pickle.dump(rdm_dict,f)

def main(args):
    get_rdms(args)

if __name__ == "__main__":
    args = parse_option()
    main(args)

    