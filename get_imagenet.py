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
import hashlib
import PIL.Image
import urllib
from bs4 import BeautifulSoup
from joblib import Parallel, delayed

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

def download_picture(url, num_images, img_path):
    if len(os.listdir(img_path)) < num_images:
        save_path = f'{img_path}/img_{hashlib.md5(url.encode()).hexdigest()}.jpg'
        #_tested.append(url)
        if os.path.exists(save_path):
            print('image already saved')
            return
        else:
            try:
                I, code = url_to_image(url)
                print(f'synset ID {synset}  response {code}')
                
                if I is None:
                    print('I is None')
                    return
                
                if len(I.shape) == 3:
                    if cv2.imwrite(save_path,I):
                        print(f'{synset} image saved successfully')
                    else:
                        print('image not saved')
                else:
                    print('image not correct size')
                    return
            
            except:
                raise
                print('Error with this url')
                return

def iter_synsets(synset, category_path):
    img_path = f'{category_path}/{synset}'
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    urls = requests.get(f'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={synset}')
    soup = BeautifulSoup(urls.content, 'html.parser')
    soup = str(soup)
    url_list = soup.split('\r\n')

    #_tested = []
    random.shuffle(url_list)
    Parallel(n_jobs=64)(delayed(download_picture)(url, num_images, img_path) for url in url_list)

Parallel(n_jobs=64)(delayed(iter_synsets)(synset, category_path) for synset in test_categs)