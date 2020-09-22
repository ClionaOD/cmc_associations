import os
import random
from shutil import copyfile

og_path = '/data/imagenet_cmc'
new_path = '/home/clionaodoherty/imagenet_test'

done = []
copyCount = 0
while copyCount < 5:
    cp_folder  = random.choice(os.listdir(og_path))
    if os.path.isdir(f'{og_path}/{cp_folder}') and not cp_folder in done:
        for i in range(10):
            f = random.choice(os.listdir(f'{og_path}/{cp_folder}'))
            src = f'{og_path}/{cp_folder}/{f}'
            dst = f'{new_path}/{cp_folder}/{f}'
            if not os.path.exists(dst):
                os.makedirs(f)
            copyfile(src, dst)
        done.append(cp_folder)