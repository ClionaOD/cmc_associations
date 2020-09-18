import argparse
import os
import scipy.io
import pickle
import time
from collections import OrderedDict

#import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import alexnet
import datetime

#import clustering
import models
from util import AverageMeter

from dataset import RGB2Lab
from models.alexnet import TemporalAlexNetCMC
from train_CMC import get_color_distortion


def parse_option():

    parser = argparse.ArgumentParser('argument for activations')

    parser.add_argument('--model_path', type=str, help='model to get activations of')
    parser.add_argument('--save_path', type=str, help='path to save activations')
    parser.add_argument('--transform', type=str, choices=['Lab','distort'], help='color transform to use')
    parser.add_argument('--supervised', type=bool, default=False, help='whether to test against supervised AlexNet')

    opt = parser.parse_args()

    return opt

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

def compute_features(dataloader, model, N):
    print('Compute features')
    model.eval()
    act = {}
    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        _model_feats.append(output.cpu().numpy())
    for m in model.modules():               #model.encoder.modules() for my own alexnet/model
        if isinstance(m, nn.ReLU):
            m.register_forward_hook(_store_feats)
    #for m in model.classifier.modules():
    #    if isinstance(m, nn.ReLU):
    #        m.register_forward_hook(_store_feats)
    for i, input_tensor in enumerate(dataloader):
        with torch.no_grad():
            input_var, label = input_tensor[0].cuda(),input_tensor[2]
            input_var = input_var.float()
            _model_feats = []
            aux = model(input_var).data.cpu().numpy()
            act[label[0]] = _model_feats
    return act

def get_activations(offset, args):
    mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
    std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
    if args.transform == 'Lab':
        color_transfer = RGB2Lab()
    else:
        color_transfer = get_color_distortion()
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        color_transfer,
        transforms.ToTensor(),
        normalize,
    ])
    dataset = ImageFolderWithPaths(offset, transform=train_transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=1,
                                            num_workers=0,
                                            pin_memory=True,
                                            shuffle = False)
    features = compute_features(dataloader, model, len(dataset))
    return features


if __name__ == '__main__':
    args = parse_option()

    if not args.supervised:
        modelpth = args.model_path
        checkpoint = torch.load(modelpth)['model']

        model = TemporalAlexNetCMC()
        model.load_state_dict(checkpoint)
        model.cuda()
    else:
        model = alexnet(pretrained=True)
        model.cuda()

    image_pth = '/home/clionaodoherty/imagenet_samples/' 
    act = get_activations(image_pth, args)
    print('activations computed')

    with open('/home/clionaodoherty/CMC/category_dict.pickle','rb') as f:
        categories= pickle.load(f)
    categories = list(categories.values())
    categories = [list(k.keys()) for k in categories]
    categories = [item for sublist in categories for item in sublist]

    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
    activations = {k:{l:[] for l in layers} for k in categories}

    act_list = list(act.items()) #this is list of tuples len 50*34, x[0] is the path x[1] is the list of acts one per layer

    for path, activation_list in act_list:
        for label in categories:
            if label in path:
                for idx, l in enumerate(layers):
                    activations[label][l].append(activation_list[idx])

    print('calculating mean activations')
    for label in categories:
        for l in layers:
            mean = activations[label][l][0]
            for i in activations[label][l][1:]:
                mean = np.concatenate((mean,i), axis=0)
            mean = np.mean(mean, axis=0)
            activations[label][l] = mean
    print('done ... saving')

    with open(args.save_path, 'wb') as handle:
        pickle.dump(activations, handle)