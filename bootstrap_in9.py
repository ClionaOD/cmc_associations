


import os
import pickle
import numpy as np
from pathlib import Path
from numpy import save

from joblib import Parallel, delayed

from scipy.stats import pearsonr
from scipy.spatial.distance import squareform

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

    del activations
    del activation_dfs

    return rdm_dict

def run_bootstrap(args, checkpoint, layer, mean, std, imagesets):
        
        # Need to load up models every iteration or else issues with memory
        if args.supervised:
            model = alexnet(pretrained=True)
            model.cuda()
        else:
            model = TemporalAlexNetCMC()
            model.load_state_dict(checkpoint)
            model.cuda()
        
        # HEAVY AND SLOW HERE
        rdm_dict_og = run_activation_analysis(imagesets[0], model, args, mean, std, layer)
        torch.cuda.empty_cache()
        rdm_dict_mr = run_activation_analysis(imagesets[1], model, args, mean, std, layer)
        torch.cuda.empty_cache()
        rdm_dict_ms = run_activation_analysis(imagesets[2], model, args, mean, std, layer)
        torch.cuda.empty_cache()
        rdm_dict_only_bg = run_activation_analysis(imagesets[3], model, args, mean, std, layer)
        torch.cuda.empty_cache()
        
        # r(original, mixed_rand)
        corr_coef_ogmr, p_val = pearsonr(squareform(rdm_dict_og[layer]), squareform(rdm_dict_mr[layer]))
        #results_og_mr[layer].append((corr_coef_ogmr,p_val))

        # r(original, mixed_same)
        corr_coef_ogms, p_val = pearsonr(squareform(rdm_dict_og[layer]), squareform(rdm_dict_ms[layer]))
        #results_og_ms[layer].append((corr_coef_ogms,p_val))

        # r(original, only_bg_t)
        corr_coef_ogonlybg, p_val = pearsonr(squareform(rdm_dict_og[layer]), squareform(rdm_dict_only_bg[layer]))
        #results_og_only_bg[layer].append((corr_coef_ogonlybg,p_val))

        #diff_ogms_less_ogmr[layer].append(corr_coef_ogms - corr_coef_ogmr)
        #diff_ogmr_less_ogonlybg[layer].append(corr_coef_ogmr - corr_coef_ogonlybg)
        
        return corr_coef_ogmr, corr_coef_ogms, corr_coef_ogonlybg

def main(args, in9_types, model_name='', n_bootstraps=1000, chosen_layers=['conv5'], save_path='/data/movie-associations/bootstrapping'):
    
    # Set mean and std depending on model
    if args.supervised:
        print(f'MODEL: supervised')

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
        
        mean=[0.4493,0.4348,0.3970] ; std=[0.3030, 0.3001, 0.3016]
    
    # First, get the activations by sampling with replacement
    # will need to do twice - one per IN-9 type so that RDMs can be correlated

    imageset_og = os.path.join(f'{args.image_path}','original','val')
    imageset_mr = os.path.join(f'{args.image_path}','mixed_rand','val')
    imageset_ms = os.path.join(f'{args.image_path}','mixed_same','val')
    imageset_only_bg = os.path.join(f'{args.image_path}','only_bg_t','val')

    imagesets = [imageset_og,imageset_mr,imageset_ms,imageset_only_bg]

    all_results = {k:[] for k in chosen_layers}
    
    for layer in chosen_layers:
        all_results[layer].extend(Parallel(n_jobs=4)(delayed(run_bootstrap)(args, checkpoint, layer, mean, std, imagesets) for B in range(n_bootstraps)))
    
    results_og_mr = {layer:np.array([bootstrap[0] for bootstrap in results]) for layer,results in all_results.items()}
    results_og_ms = {layer:np.array([bootstrap[1] for bootstrap in results]) for layer,results in all_results.items()}
    results_og_onlybg = {layer:np.array([bootstrap[2] for bootstrap in results]) for layer,results in all_results.items()}

    diff_ogms_less_ogmr = {k:results_og_ms[k] - results_og_mr[k] for k in chosen_layers}
    diff_ogmr_less_ogonlybg = {k:results_og_mr[k] - results_og_onlybg[k] for k in chosen_layers}
    
    # Save the three correlations
    Path(f'{save_path}/original_mixed_rand').mkdir(parents=True, exist_ok=True)
    with open(f'{save_path}/original_mixed_rand/{model_name.split("_")[0]}_exemplar-bootstrap.pickle', 'wb') as f:
        pickle.dump(results_og_mr,f)

    Path(f'{save_path}/original_mixed_same').mkdir(parents=True, exist_ok=True)
    with open(f'{save_path}/original_mixed_same/{model_name.split("_")[0]}_exemplar-bootstrap.pickle', 'wb') as f:
        pickle.dump(results_og_ms,f)

    Path(f'{save_path}/original_only_bg').mkdir(parents=True, exist_ok=True)
    with open(f'{save_path}/original_only_bg/{model_name.split("_")[0]}_exemplar-bootstrap.pickle', 'wb') as f:
        pickle.dump(results_og_onlybg,f)
    
    # Save the two differences
    Path(f'{save_path}/og_ms_less_og_mr').mkdir(parents=True, exist_ok=True)
    with open(f'{save_path}/og_ms_less_og_mr/{model_name.split("_")[0]}_exemplar-bootstrap.pickle', 'wb') as f:
        pickle.dump(diff_ogms_less_ogmr,f)
    
    Path(f'{save_path}/og_mr_less_ogonlybg').mkdir(parents=True, exist_ok=True)
    with open(f'{save_path}/og_mr_less_ogonlybg/{model_name.split("_")[0]}_exemplar-bootstrap.pickle', 'wb') as f:
        pickle.dump(diff_ogmr_less_ogonlybg,f)

if __name__ == '__main__':
    args = parse_option()

    # for this script, image path is just the root of the IN-9 testset
    args.model_path = '/data/movie-associations/weights_for_eval/main'
    args.image_path = '/data/movie-associations/bg_challenge'

    args.dataset = 'bg_challenge'

    args.model = 'alexnet'

    args.supervised = False

    args.exemplar_bootstrapping = True

    og = 'original'
    mr = 'mixed_rand'
    ms = 'mixed_same'
    only_bg = 'only_bg_t'

    in9_types=[og,mr,ms,only_bg]

    B = 1000

    for model in os.listdir(args.model_path):
        main(args, in9_types, model, B)
    
    # then do supervised
    args.supervised=True
    main(args, in9_types, model_name='supervised', n_bootstraps=B)