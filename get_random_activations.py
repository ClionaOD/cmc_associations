from shutil import Error
from urllib.request import FileHandler
from torch._C import FileCheck
from get_activations import compute_features, parse_option, get_color_distortion, get_activations
import torchvision.transforms as transforms
from dataset import RGB2Lab
import os
import torch
from models.alexnet import TemporalAlexNetCMC
from torchvision.models import alexnet
import pickle

def main(args, model_weights=''):
    modelpth = os.path.join(args.model_path, model_weights)
    print(f'MODEL: {model_weights}')

    _file = model_weights.split('_')[0]
    if not _file == 'random':
        raise Error('The weights provided are not random please check paths')
    
    modelpth = os.path.join(args.model_path, model_weights)

    # Only need this syntax because will load random only
    checkpoint = torch.load(modelpth)

    model = TemporalAlexNetCMC()
    model.load_state_dict(checkpoint)
    model.cuda()

    if not args.supervised:
        if args.dataset == 'mscoco':
            activations = get_activations(args.image_path, model, args, mean=[0.4240,0.4082,0.3853], std=[0.2788, 0.2748, 0.2759])
        else:
            activations = get_activations(args.image_path, model, args, mean=[0.4493,0.4348,0.3970], std=[0.3030, 0.3001, 0.3016])
    else:
        activations = get_activations(args.image_path, model, args)

    print('done ... saving')

    if not args.supervised:
        _save = f'{args.save_path}/{_file}-{args.transform}_activations.pickle'
        if args.blur:
            _save = f'{args.save_path}/{_file}-{args.transform}_blur_sigma{args.sigma}_kernel{args.kernel_size}_activations.pickle'
    else:
        _save = f'{args.save_path}/{_file}-supervised_activations.pickle'
        if args.blur:
            _save = f'{args.save_path}/{_file}-supervised_blur_sigma{args.sigma}_kernel{args.kernel_size}_activations.pickle'
    
    with open(_save, 'wb') as handle:
        pickle.dump(activations, handle)

if __name__ == '__main__':
    args = parse_option()

    # args.model_path = '/data/movie-associations/weights_for_eval/main_replic'
    # args.image_path = '/data/movie-associations/bg_challenge/original/val'
    # args.save_path = '/data/movie-associations/activations/bg_challenge/replic_training/original'

    # args.dataset = 'bg_challenge'

    colors = ['distort', 'Lab', 'distort'] # [-1] is distort because default but this won't be used
    model = 'random_alexnet.pth'
    sup = [False, False, True]
    
    for i in range(3):
        args.transform = colors[i]
        args.supervised = sup[i]
        
        main(args, model)