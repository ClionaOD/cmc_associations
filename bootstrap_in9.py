import os
import pickle
import numpy as np
import pandas as pd
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

def run_activation_bootstrap(activations, boot_inds, chosen_layers):
    # TODO: let the indexing work if array isn't 4D
    activations = {category:{layer:np.mean(a[boot_inds,:,:,:],axis=0) for layer,a in layers.items() if layer in chosen_layers} for category,layers in activations.items()}
    activation_dfs = construct_activation_df(activations, meaned=True)
    rdm_dict = {k:construct_rdm(v) for k,v in activation_dfs.items() if k in chosen_layers}

    return rdm_dict

def bootstrap(og_activations, mr_activations, onlybg_activations, og_rdm_dict_no_bootstrap, layer):
    boot_inds = np.random.randint(450, size=450)
            
    rdm_dict_og=run_activation_bootstrap(og_activations, boot_inds, [layer])
    rdm_dict_mr=run_activation_bootstrap(mr_activations, boot_inds, [layer])
    rdm_dict_onlybg = run_activation_bootstrap(onlybg_activations, boot_inds, [layer])

    # r(original, mixed_rand)
    corr_coef_ogmr, p_val = pearsonr(squareform(rdm_dict_og[layer]), squareform(rdm_dict_mr[layer]))

    # # r(original, only_bg_t)
    corr_coef_ogonlybg, p_val = pearsonr(squareform(rdm_dict_og[layer]), squareform(rdm_dict_onlybg[layer]))

    # r(original,original) - baseline
    corr_coef_ogog, p_val = pearsonr(squareform(og_rdm_dict_no_bootstrap[layer]), squareform(rdm_dict_og[layer]))

    return corr_coef_ogmr, corr_coef_ogonlybg, corr_coef_ogog

def main(args, in9_types, model_name='', n_bootstraps=1000, chosen_layers=['conv5'], save_path='/data/movie-associations/bootstrapping', training='main'):
    
    _modelp = model_name.split('_')[0]
    print(f'MODEL: {_modelp}')

    acts = []
    for in9_type in in9_types[:1]:
        type_act_path = os.path.join('/data/movie-associations/activations/bg_challenge',training,in9_type,'all',f'{_modelp}_activations.pickle')
        if os.path.exists(type_act_path):
            calc_acts=False
        else:
            calc_acts=True
    
        if calc_acts:
            # Set mean and std depending on model
            if args.supervised:
                model = alexnet(pretrained=True)
                model.cuda()

                checkpoint = ''

                mean=[(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
                std=[(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
            else:
                # Load up this model
                modelpth = os.path.join(args.model_path, model_name)
                
                if args.model == 'alexnet':
                    model = TemporalAlexNetCMC()
                
                
                if 'finetune' in modelpth or 'resnet' in modelpth:
                    checkpoint = torch.load(modelpth)['model']
                else:
                    checkpoint = torch.load(modelpth)
                
                model.load_state_dict(checkpoint)
                model.cuda()
                
                if 'lab' in modelpth.lower():
                    args.transform = 'Lab'
                    mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
                    std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
                else:
                    args.transform = 'distort'
                    mean=[0.4493,0.4348,0.3970]
                    std=[0.3030, 0.3001, 0.3016]
                

            imageset = os.path.join(f'{args.image_path}',in9_type,'val')

            acts.append(get_activations(imageset, model, args, mean=mean, std=std, calc_mean=False))
            
            torch.cuda.empty_cache()
            
            # Path(os.path.join(args.save_path,training,in9_type,'all')).mkdir(parents=True, exist_ok=True)
            # with open(type_act_path,'wb') as f:
            #     pickle.dump(activations,f)
            
            # del activations
        else:
            acts.append(pd.read_pickle(os.path.join(args.save_path,training,in9_type,'all',f'{_modelp}_activations.pickle')))
    

    og_activations = acts[0]
    mr_activations = acts[1]
    onlybg_activations = acts[2]
    

    all_results = {k:[] for k in chosen_layers}
    
    for layer in chosen_layers:
        # do actual og_activations without bootstrap to get comparator
        og_rdm_dict_no_bootstrap = run_activation_bootstrap(og_activations, list(range(450)), [layer])

        all_results[layer].extend(Parallel(n_jobs=4)(delayed(bootstrap)(og_activations,mr_activations,onlybg_activations,og_rdm_dict_no_bootstrap,layer) for B in range(n_bootstraps)))
        
        for B in range(n_bootstraps):
            # boot_inds = np.random.randint(450, size=450)
            
            # rdm_dict_og=run_activation_bootstrap(og_activations, boot_inds, [layer])
            # rdm_dict_mr=run_activation_bootstrap(mr_activations, boot_inds, [layer])
            # rdm_dict_onlybg = run_activation_bootstrap(onlybg_activations, boot_inds, [layer])

            # # r(original, mixed_rand)
            # corr_coef_ogmr, p_val = pearsonr(squareform(rdm_dict_og[layer]), squareform(rdm_dict_mr[layer]))

            # # # r(original, only_bg_t)
            # corr_coef_ogonlybg, p_val = pearsonr(squareform(rdm_dict_og[layer]), squareform(rdm_dict_onlybg[layer]))

            # # r(original,original) - baseline
            # corr_coef_ogog, p_val = pearsonr(squareform(og_rdm_dict_no_bootstrap[layer]), squareform(rdm_dict_og[layer]))
            
            # all_results[layer][0].append(corr_coef_ogmr)
            # all_results[layer][1].append(corr_coef_ogonlybg)
            # all_results[layer][2].append(corr_coef_ogog)


    
    results_og_mr = {layer:np.array(results[0]) for layer,results in all_results.items()}
    results_og_onlybg = {layer:np.array(results[1]) for layer,results in all_results.items()}
    results_og_og = {layer:np.array(results[2]) for layer,results in all_results.items()}

    diff_ogmr_less_ogonlybg = {k:results_og_mr[k] - results_og_onlybg[k] for k in chosen_layers}
    
    # Save the two correlations and the difference
    Path(f'{save_path}/original_mixed_rand').mkdir(parents=True, exist_ok=True)
    with open(f'{save_path}/original_mixed_rand/{_modelp}_exemplar-bootstrap.pickle', 'wb') as f:
        pickle.dump(results_og_mr,f)

    Path(f'{save_path}/original_only_bg').mkdir(parents=True, exist_ok=True)
    with open(f'{save_path}/original_only_bg/{_modelp}_exemplar-bootstrap.pickle', 'wb') as f:
        pickle.dump(results_og_onlybg,f)
    
    Path(f'{save_path}/original_original').mkdir(parents=True, exist_ok=True)
    with open(f'{save_path}/original_original/{_modelp}_exemplar-bootstrap.pickle', 'wb') as f:
        pickle.dump(results_og_og,f)
    
    Path(f'{save_path}/og_mr_less_ogonlybg').mkdir(parents=True, exist_ok=True)
    with open(f'{save_path}/og_mr_less_ogonlybg/{_modelp}_exemplar-bootstrap.pickle', 'wb') as f:
        pickle.dump(diff_ogmr_less_ogonlybg,f)

if __name__ == '__main__':
    args = parse_option()

    # for this script, image path is just the root of the IN-9 testset
    args.model_path = '/data/movie-associations/weights_for_eval/main'
    args.image_path = '/data/movie-associations/bg_challenge'
    args.save_path = '/data/movie-associations/activations/bg_challenge'

    args.dataset = 'bg_challenge'

    args.model = 'alexnet'

    args.supervised = False

    og = 'original'
    mr = 'mixed_rand'
    ms = 'mixed_same'
    only_bg = 'only_bg_t'

    in9_types=[og,mr,only_bg]

    B = 2

    for model in os.listdir(args.model_path)[4:]:
        main(args, in9_types, model, B)
    
    # then do supervised
    args.supervised=True
    main(args, in9_types, model_name='supervised', n_bootstraps=B)