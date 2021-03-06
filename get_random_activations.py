from shutil import Error
from urllib.request import FileHandler
from torch._C import FileCheck
from get_activations import compute_features, parse_option, get_color_distortion, get_activations
import torchvision.transforms as transforms
from dataset import RGB2Lab
import os
import torch
from torchvision.models import alexnet
from models.alexnet import TemporalAlexNetCMC
import pickle

def main(args, model_weights=''):
    modelpth = os.path.join(args.model_path, model_weights)
    print(f'MODEL: {model_weights}')

    _file = model_weights.split('_')[0]
    if not _file == 'random':
        raise Error('The weights provided are not random please check paths')
    
    if not args.supervised:
        print('not supervised')
        modelpth = os.path.join(args.model_path, model_weights)

        # Only need this syntax because will load random only
        checkpoint = torch.load(modelpth)

        model = TemporalAlexNetCMC()
        model.load_state_dict(checkpoint)
        model.cuda()

        activations = get_activations(args.image_path, model, args)
    else:
        print('supervised')
        model = alexnet(pretrained=True)
        model.cuda()

        activations = get_activations(args.image_path, model, args, mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])

    print('done ... saving')

    if not args.supervised:
        _save = f'{args.save_path}/{_file}_{args.transform}_activations.pickle'
        if args.blur:
            _save = f'{args.save_path}/{_file}_blur_sigma{args.sigma}_kernel{args.kernel_size}_activations.pickle'
    else:
        _save = f'{args.save_path}/{_file}_supervised_activations.pickle'
        if args.blur:
            _save = f'{args.save_path}/supervised_blur_sigma{args.sigma}_kernel{args.kernel_size}_activations.pickle'
    
    with open(_save, 'wb') as handle:
        pickle.dump(activations, handle)

if __name__ == '__main__':
    args = parse_option()

    args.model_path = '/data/movie-associations/weights_for_eval/main'
    args.save_path = '/data/movie-associations/activations...'
    args.image_path = '/data/movie-associations/imagenet_cmc_256/to_test'

    colors = ['distort', 'Lab', 'distort'] # [-1] is distort because default but this won't be used
    models = ['random_alexnet.pth', 'random_alexnet.pth', '']
    sup = [False, False, True]
    
    for i in range(3):
        args.transform = colors[i]
        args.supervised = sup[i]
        
        main(args, models[0])