import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
import math
import unittest
from numpy.lib.arraysetops import in1d
from numpy.lib.npyio import save

import torch
import torch.fft #for fourier
import torch.nn as nn
from torch.nn.modules import activation
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import alexnet

from dataset import RGB2Lab
from models.alexnet import TemporalAlexNetCMC
from PIL import Image
from segmentation import segment_vals, segment_with_fourier

def parse_option(): 

    parser = argparse.ArgumentParser('argument for activations')

    parser.add_argument('--model_path', type=str, help='model to get activations of')
    parser.add_argument('--save_path', type=str,help='path to save activations')
    parser.add_argument('--image_path', type=str, help='path to images for activation analysis')
   
    parser.add_argument('--transform', type=str, default='distort', choices=['Lab','distort'], help='color transform to use')
   
    parser.add_argument('--supervised', type=bool, default=False, help='whether to test against supervised AlexNet')
   
    parser.add_argument('--segment', type=str, default=None, choices=['rm_bg','rm_obj'], help='whether to segment the objects from bg and which to remove')
    
    parser.add_argument('--blur', type=bool, default=False, help='if not None, this will blur the image using a Gaussian kernel with sigma and kernel defined below')
    parser.add_argument('--sigma', type=float, default=10.0, help='sigma size for blurring if args.blur is not None')
    parser.add_argument('--kernel_size', type=int, default=15, help='paramater for setting the gaussian kernel size if this is preferred for blurring')

    opt = parser.parse_args()

    return opt

def get_color_distortion(s=1.0):
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter],p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def compute_features(dataloader, model, categories, layers):
    print('Compute features')
    model.eval()
    
    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        _model_feats.append(output.cpu().numpy())
    for m in model.modules():               #model.encoder.modules() for my own alexnet/model
        if isinstance(m, nn.ReLU):
            m.register_forward_hook(_store_feats)
    
    activations = {categ:{l:[] for l in layers} for categ in categories}
    #categ is the n0XXX imagenet class - n_categs = 256 (len activations = 256)
    #l is the layer ('conv1' etc.) - running avg of acts, sum with each then divide by 150 (150 imgs per class)

    print('... working on activations ...')
    
    #choose random indices to save examples
    save_idx = random.sample(range(len(dataloader)), k=15)
    save_idx.append(0)
    
    for i, input_tensor in enumerate(dataloader):  
        with torch.no_grad():
            
            input_var, label = input_tensor[0].cuda(),input_tensor[2][0]
            #TODO: check / make less specific
            category = label.split('/')[-2]
            if i in save_idx:
                imsave(input_var[0,:,:,:].cpu(),title=f'./imgs/rm_bg/img{i}_input.png')
            
            if args.segment == 'rm_bg':
                
                input_var, scramb = segment_with_fourier(
                    input_var, 
                    inp_mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2],
                    inp_std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2],
                    #inp_mean=[0.4493, 0.4348, 0.3970], 
                    #inp_std=[0.3030, 0.3001, 0.3016], 
                    remove='background',
                    model='dlab_pascal')
                
                input_var = input_var.unsqueeze(0)
                
                if i in save_idx:
                    imsave(input_var[0,:,:,:],title=f'./imgs/rm_bg/img{i}_segmented.png')
                    imsave(scramb,title=f'./imgs/rm_bg/img{i}_scrambled.png')
            
            elif args.segment == 'rm_obj':
                input_var, scramb = segment_with_fourier(
                    input_var, 
                    inp_mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2],
                    inp_std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2],
                    #inp_mean=[0.4493, 0.4348, 0.3970], 
                    #inp_std=[0.3030, 0.3001, 0.3016],  
                    remove='objects',
                    model='dlab_pascal')
                input_var = input_var.unsqueeze(0)
                
                #_img_sample_path = './imgs/rm_obj'
                #if i in save_idx and not len(os.listdir(_img_sample_path))==15:
                #    imsave(input_var[0,:,:,:],title=f'{_img_sample_path}/seg_obj_img_{i}.png')
            
            input_var = input_var.float().cuda()
            _model_feats = []
            model(input_var)
            
            if i == 0:
                zero_arrs = [np.zeros(arr.shape) for arr in _model_feats]
                activations = {categ:{l:zero_arrs[idx] for idx, l in enumerate(layers)} for categ in categories}

            for idx, acts in enumerate(_model_feats): 
                activations[category][layers[idx]] = activations[category][layers[idx]] + acts
            
    
    print('... getting mean ...')
    for categ in categories:
        for layer in layers:
            activations[categ][layer] = (activations[categ][layer] / 150)
            #150 is because there are 256 * 150 images in the test set, need to make an argument
            #remove first dimension for batch size

    return activations

def imsave(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    
    #original CMC mean/std for comparison
    mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
    std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
    
    #vals returned from get_mean_std.py
    #mean = [0.4493, 0.4348, 0.3970]
    #std = [0.3030, 0.3001, 0.3016]
    
    mean = np.array(mean)
    std = np.array(std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imsave(title, inp)

def get_activations(imgPath, model, args, mean=[(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2], std=[(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]):
    
    #transform the input images
    #vals returned from get_mean_std.py
    #mean = [0.4493, 0.4348, 0.3970]
    #std = [0.3030, 0.3001, 0.3016]

    #original CMC mean/std for comparison
    mean = mean
    std = std
    normalize = transforms.Normalize(mean=mean, std=std)
    gauss = transforms.GaussianBlur(kernel_size=(args.kernel_size,args.kernel_size), sigma=(args.sigma,args.sigma))
    
    if not args.supervised:   
        if args.transform == 'Lab':
            color_transfer = RGB2Lab()
        else:
            color_transfer = get_color_distortion()

        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            color_transfer,
            transforms.ToTensor(),
            normalize
        ])
        
        if args.blur:
            train_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                color_transfer,
                transforms.ToTensor(),
                gauss,
                normalize
            ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        
        if args.blur:
            train_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                gauss,
                normalize
            ])
    
    #load the data
    dataset = ImageFolderWithPaths(imgPath, transform=train_transform)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=1,
                                            num_workers=0,
                                            pin_memory=True,
                                            shuffle = False)
    
    #TODO: check/make less specific
    categories = []
    for d in os.listdir(args.image_path):
        if os.path.isdir(f'{args.image_path}/{d}'):
            categories.append(d)

    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
    
    #compute the features
    mean_activations = compute_features(dataloader, model, categories=categories, layers=layers)
    
    return mean_activations

def main(args, model_weights=''):
    modelpth = os.path.join(args.model_path, model_weights)
    print(f'MODEL: {model_weights}')
    
    if not args.supervised:
        print('not supervised')
        modelpth = os.path.join(args.model_path, model_weights)

        if 'lab' in modelpth.lower():
            args.transform = 'Lab'

        if 'finetune' in modelpth:
            checkpoint = torch.load(modelpth)['model']
        else:
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
        _file = model_weights.split('_')[0]
        _save = f'{args.save_path}/{_file}_activations.pickle'
        if args.blur:
            _save = f'{args.save_path}/{_file}_blur_sigma{args.sigma}_kernel{args.kernel_size}_activations.pickle'
    else:
        _save = f'{args.save_path}/supervised_activations.pickle'
        if args.blur:
            _save = f'{args.save_path}/supervised_blur_sigma{args.sigma}_kernel{args.kernel_size}_activations.pickle'
    
    with open(_save, 'wb') as handle:
        pickle.dump(activations, handle)

if __name__ == '__main__':
    args = parse_option()
    print('args parsed')

    if not args.supervised:
        models = [m for m in os.listdir(args.model_path) if '.pth' in m]
        for m in models:
            main(args,m)
    else:
        main(args)