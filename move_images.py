import os
import shutil

base_path = '/data/imagenet_cmc'

wnids = os.listdir(base_path)

for wnid in wnids:
    if len(os.path.join(base_path,wnid)) == 150:
        source = os.path.join(base_path,wnid)
        dest = os.path.join(base_path,'to_test',wnid)
        if not os.path.exists(dest):
            os.makedirs(dest)
        shutil.move(source,dest)