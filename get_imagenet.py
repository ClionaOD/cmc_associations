# list of synsets at following API http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list
# image URLs by synset ID at http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=[wnid]
# map image names to image urls using http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid=[wnid]

#TODO: get list of imagenet synsets, random sample 500 synsets
    #save this list to re-use for replicability (if check)

#TODO: get 150 images per class
    # check if already saved on server, if then don't redownload else get and save

import requests
import json
import random
import os
import numpy as np
import cv2
import PIL.Image
import urllib
from bs4 import BeautifulSoup

num_categories = 500
num_images = 150

category_path = f'/data/imagenet_cmc'
#category_path = f'/home/clionaodoherty/Desktop/imagenet_categories'
list_path = f'{category_path}/imagenet_categs_{num_categories}.json'
test_categs = []

#TODO: make this a way to extend a list and not make it an entirely new list
if os.path.exists(list_path):
    with open(list_path, 'r') as f:
        test_categs = json.load(f) 
else:
    synsets = requests.get('http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list').iter_lines()
    synset_list = [str(i)[2:-1] for i in synsets]

    for i in range(num_categories):
        s = random.choice(synset_list)
        if len(s) == 9:
            test_categs.append(s)

    with open(f'{list_path}','w') as f:
        json.dump(test_categs,f)

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read it into OpenCV format
    resp = urllib.request.urlopen(url)
    code = resp.getcode()
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image, code

for synset in test_categs:
    img_path = f'{category_path}/{synset}'
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    urls = requests.get(f'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={synset}')
    soup = BeautifulSoup(urls.content, 'html.parser')
    soup = str(soup)
    url_list = soup.split('\r\n')
    
    #TODO: use a generator to do next for saving the images
    _tested = []
    while len(os.listdir(img_path)) < num_images and not len(_tested) == len(url_list):
        _url = random.choice(url_list)
        if not _url in _tested:
            save_path = f'{img_path}/img_{len(os.listdir(img_path))}.jpg'
            _tested.append(_url)
            if os.path.exists(save_path):
                print('image already saved')
            else:
                try:
                    I, code = url_to_image(_url)
                    print(f'synset ID {synset}  response {code}')
                    if I == None:
                        print('I is None')
                    if cv2.imwrite(save_path,I):
                        print(f'{synset} image saved successfully')
                    else:
                        print('image not saved')
                except urllib.error.HTTPError:
                    print('HTTP Error')
                    