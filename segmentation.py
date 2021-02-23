from torch._C import autocast_increment_nesting
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.hub
import torch.fft
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd

dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

#COCO Stuff trained model
dlab_coco = torch.hub.load("kazuto1011/deeplab-pytorch", "deeplabv2_resnet101", pretrained='cocostuff164k', n_classes=182).eval()

class NormalizeInverse(T.Normalize):
  """
  Undoes the normalization and returns the reconstructed images in the input domain.
  """

  def __init__(self, mean, std):
      mean = torch.as_tensor(mean)
      std = torch.as_tensor(std)
      std_inv = 1 / (std + 1e-7)
      mean_inv = -mean * std_inv
      super().__init__(mean=mean_inv, std=std_inv)

  def __call__(self, tensor):
      return super().__call__(tensor.clone())        

# Function to place objects on a white background
def segment_objects(segmap, img, nc=21):
  
  #These are the predefined label indices for Pascal-VOC
  #Keep here in case a specific tag is needed for future work

  label_colors = np.array([(255, 255, 255),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.where(segmap > 0, img[:,:,0], segmap)
  g = np.where(segmap > 0, img[:,:,1], segmap)
  b = np.where(segmap > 0, img[:,:,2], segmap)

  for l in range(0, nc):
    idx = segmap == l
    if l == 0: 
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

  rgb = np.stack([r, g, b], axis=2)
  return rgb

# Function to remove objects from background, leaving the background
def segment_background(seg_img, transf_img, nc=21):
  
  label_colors = np.array([(255, 255, 255),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.where(seg_img == 0, transf_img[:,:,0], label_colors[0,0])
  g = np.where(seg_img == 0, transf_img[:,:,1], label_colors[0,1])
  b = np.where(seg_img == 0, transf_img[:,:,2], label_colors[0,2])

  rgb = np.stack([r, g, b], axis=2)
  return rgb

def segment(net, path):
  img = Image.open(path)
  
  # Use ImageNet mean and std
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  
  # Resize and CenterCrop for better inference results
  trf = T.Compose([T.Resize(256), 
                    T.CenterCrop(224), 
                    T.ToTensor(), 
                    T.Normalize(mean = mean, 
                                std = std)])
  
  inp = trf(img).unsqueeze(0)
  
  transf_r = inp[0,0,:,:].numpy() ; transf_g = inp[0,1,:,:].numpy(); transf_b = inp[0,2,:,:].numpy()
  transf_img = np.stack([transf_r, transf_g, transf_b], axis=0)
  unorm = NormalizeInverse(mean=mean, std=std)
  transf_img = unorm(torch.from_numpy(transf_img)).numpy()
  transf_img = np.stack([transf_img[0,:,:], transf_img[1,:,:], transf_img[2,:,:]], axis=2)
  plt.imshow(transf_img); plt.axis('off'); plt.savefig('transf_img.png')
  
  out = net(inp)['out']
  om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

  obj = segment_objects(om, transf_img)
  plt.imshow(obj); plt.axis('off'); plt.savefig('segmented_obj.png')

  bg = segment_background(om, transf_img)
  plt.imshow(bg); plt.axis('off'); plt.savefig('segmented_bg.png')

def phase_scramble(inp_tensor):
  inp_fft = torch.fft.rfftn(inp_tensor, dim=(2,3))
  
  new_phase=np.random.uniform(-math.pi, math.pi, size=inp_fft.size()[2:])
  inp_scramb = inp_fft * np.exp(1j*new_phase)
  
  inp_scramb = torch.fft.irfftn(inp_scramb, dim=(2,3))
  return inp_scramb

def segment_with_fourier(inp, inp_mean, inp_std, model, remove='background'):
  """must specify net which is either 'dlab_pascal' or 'dlab_coco'"""
  
  # restructure the tensor for de-normalization and reconstruct 
  img_unorm = NormalizeInverse(mean=inp_mean, std=inp_std)(inp)

  # Use ImageNet mean and std for segmentation
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  
  trf = T.Compose([T.Normalize(mean = mean, 
                               std = std)])
  
  img_seg_norm = trf(img_unorm).cpu() #this is the image normalized to segmentation (imgnet) mean and std
  
  # Get segmentation map using Google DeepLab
  if model == 'dlab_pascal':
    net = dlab
    out = net(img_seg_norm)['out']
    segmap = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  elif model == 'dlab_coco':
    net = dlab_coco
    out = net(img_seg_norm)
    _, _, H, W = img_seg_norm.shape
    out = F.interpolate(out, size=(H,W), mode="bilinear", align_corners=False)
    segmap = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

    # Get stuff and thing classes for coco-stuff segmentation
    coco_df = pd.read_csv('./cocostuff_labels.csv', index_col=0)
    coco_types = coco_df['type'].to_dict()

    segmap_types = {k:v for k,v in coco_types.items() if k in segmap}
    stuff_idxs = [k for k,v in segmap_types.items() if v=='stuff' or v=='unlabeled']
    thing_idxs = [k for k,v in segmap_types.items() if v=='thing']

    #TODO: convert to zero/non-zero for stuff/things or bg

  # Phase scramble input image tensor for replacement
  img_scrambled = phase_scramble(img_seg_norm)

  # Convert both images (img and scrambled) to RGB for segmentation mapping
  rgb_img = img_seg_norm[0,:,:,:].numpy().transpose(1,2,0)
  rgb_scramb = img_scrambled[0,:,:,:].numpy().transpose(1,2,0)

  if remove == 'background':
    r = np.where(segmap == 0, rgb_img[:,:,0], rgb_scramb[:,:,0])
    g = np.where(segmap == 0, rgb_img[:,:,1], rgb_scramb[:,:,1])
    b = np.where(segmap == 0, rgb_img[:,:,2], rgb_scramb[:,:,2])
    rgb = np.stack([r, g, b], axis=2)
  elif remove == 'objects':
    r = np.where(segmap != 0, rgb_img[:,:,0], rgb_scramb[:,:,0])
    g = np.where(segmap != 0, rgb_img[:,:,1], rgb_scramb[:,:,1])
    b = np.where(segmap != 0, rgb_img[:,:,2], rgb_scramb[:,:,2])
    rgb = np.stack([r, g, b], axis=2)
  else:
    raise ValueError('please select either background or objects to remove')
  
  rgb = rgb.transpose(2,0,1) ; rgb_scramb = rgb_scramb.transpose(2,0,1)
  rgb = torch.from_numpy(rgb) ; rgb_scramb = torch.from_numpy(rgb_scramb)
  
  rgb_unorm = NormalizeInverse(mean=mean, std=std)(rgb) ; rgb_scramb_unorm = NormalizeInverse(mean=mean, std=std)(rgb_scramb)
  rgb_inp_norm = T.Normalize(mean=inp_mean, std=inp_std)(rgb_unorm) ; rgb_scramb_inp_norm = T.Normalize(mean=inp_mean, std=inp_std)(rgb_scramb_unorm)
  
  return rgb_inp_norm , rgb_scramb_inp_norm

def segment_vals(inp, inp_mean, inp_std, net=dlab, remove='background'):
  """ 
  Function to take a tensor (from dataloader in get_activations) and return as segmented tensor
  Segmentation uses Google DeepLab v3 which requires normalization using imagenet mean and std
  Thus, the input tensor must be denormalized from whatever its transform is
  in is torch.Size([1,3,224,224]) or whatever h*w are
  """
  # restructure the tensory for de-normalization and reconstruct 
  img_unorm = NormalizeInverse(mean=inp_mean, std=inp_std)(inp)

  # Use ImageNet mean and std for segmentation
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  
  trf = T.Compose([T.Normalize(mean = mean, 
                               std = std)])
  
  img_seg_norm = trf(img_unorm).cpu()

  out = net(img_seg_norm)['out']
  segmap = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  
  #reshape torch tensor for segmentation
  rgb_img = img_seg_norm[0,:,:,:].numpy().transpose(1,2,0)

  if remove=='background':
    #place objects on white background
    obj = segment_objects(segmap, rgb_img)
    
    obj = obj.transpose(2,0,1)
    obj = torch.from_numpy(obj)
    
    obj_unorm = NormalizeInverse(mean=mean, std=std)(obj)
    obj_inp_norm = T.Normalize(mean=inp_mean, std=inp_std)(obj_unorm)
    return obj_inp_norm
  elif remove=='objects':
    #replace objects with white, show only background
    bg = segment_background(segmap, rgb_img)
    
    bg = bg.transpose(2,0,1)
    bg = torch.from_numpy(bg)
    
    bg_unorm = NormalizeInverse(mean=mean, std=std)(bg)
    bg_inp_norm = T.Normalize(mean=inp_mean, std=inp_std)(bg_unorm)
    return bg_inp_norm
  else:
    raise ValueError('not supported, please select either background or objects')

if __name__ == '__main__':
  segment(dlab, '/home/clionaodoherty/cmc_associations/test_imagenet/n10249950/img_4bf5b8c3f7c92e16a2fa26f98beda820.jpg')

