


import os
import pickle

from scipy.stats import pearsonr

import torch
import torchvision.transforms as transforms
from torchvision.models import alexnet

from dataset import RGB2Lab
from models.alexnet import TemporalAlexNetCMC

from get_activations import compute_features, get_activations, parse_option, get_color_distortion, get_activations
from get_rdms import construct_activation_df, construct_rdm

def run_activation_analysis(image_path, model, args, mean, std, chosen_layers):
    activations = get_activations(image_path, model, args, mean=mean, std=std)
    activation_dfs = construct_activation_df(activations)
    rdm_dict = {k:construct_rdm(v) for k,v in activation_dfs.items() if k in chosen_layers}
    return rdm_dict

def main(args, in9_first='original', in9_second='', model_name='', n_bootstraps=5, chosen_layers=['conv5']):
    
    
    # Set mean and std depending on model
    if args.supervised:
        print(f'MODEL: supervised')

        model = alexnet(pretrained=True)
        model.cuda()

        mean=[(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2] ; std=[(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
    else:
        # Load up this model
        modelpth = os.path.join(args.model_path, model_name)
        print(f'MODEL: {model_name}')
        
        if 'finetune' in modelpth or 'resnet' in modelpth:
            checkpoint = torch.load(modelpth)['model']
        else:
            checkpoint = torch.load(modelpth)
        
        if 'lab' in modelpth.lower():
            args.transform = 'Lab'
        else:
            args.transform = 'distort'
        
        model = TemporalAlexNetCMC()
        model.load_state_dict(checkpoint)
        model.cuda()
        
        mean=[0.4493,0.4348,0.3970] ; std=[0.3030, 0.3001, 0.3016]
    
    # First, get the activations by sampling with replacement
    # will need to do twice - one per IN-9 type so that RDMs can be correlated

    imageset_1 = os.path.join(f'{args.image_path}',in9_first,'val')
    imageset_2 = os.path.join(f'{args.image_path}',in9_second,'val')

    results = {k:[] for k in chosen_layers}
    for i in range(n_bootstraps):
        # HEAVY AND SLOW HERE
        rdm_dict_1 = run_activation_analysis(imageset_1, model, args, mean, std, chosen_layers)
        rdm_dict_2 = run_activation_analysis(imageset_1, model, args, mean, std, chosen_layers)

        for layer in chosen_layers:
            # INCORRECT BUT DRAFTING
            corr_coef, p_val = pearsonr(rdm_dict_1[layer],rdm_dict_2[layer])
            results[layer].append((corr_coef,p_val))

    return

if __name__ == '__main__':
    args = parse_option()

    # for this script, image path is just the root of the IN-9 testset
    args.model_path = '/data/movie-associations/weights_for_eval/main'
    args.image_path = '/data/movie-associations/bg_challenge'

    args.dataset = 'bg_challenge'

    args.model = 'alexnet'

    args.supervised = False

    args.exemplar_bootstrapping = True