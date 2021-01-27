import torch
import torchvision.transforms as transforms
from get_activations import get_color_distortion, ImageFolderWithPaths

def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _, _ in loader:

        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

def load_data(imgPath):
    color_transfer = get_color_distortion()

    train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            color_transfer,
            transforms.ToTensor()
        ])
        
    #load the data
    dataset = ImageFolderWithPaths(imgPath, transform=train_transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=64,
                                            num_workers=0,
                                            pin_memory=True,
                                            shuffle = False)
    return dataloader

if __name__ == '__main__':
    imgPath = '/data/imagenet_cmc/to_test'

    loader = load_data(imgPath)

    eval_mean,eval_std = online_mean_and_sd(loader)
    print(f'mean:{eval_mean} \n std:{eval_std}')