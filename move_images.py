import os
import shutil
import random 

base_path = '/data/imagenet_cmc'

wnids = os.listdir(base_path)
wnids = [wnid for wnid in wnids if not 'json' in wnid and not 'to_test' in wnid]
wnids = [wnid for wnid in wnids if len(os.listdir(os.path.join(base_path,wnid))) == 150]
mv_wnids = []
if len(wnids) > 256:
    while len(mv_wnids) < 256:
        choice = random.choice(wnids)
        if not choice in mv_wnids:
            mv_wnids.append(choice)

for wnid in mv_wnids:
    source = os.path.join(base_path,wnid)
    dest = os.path.join(base_path,'to_test')
    shutil.move(source,dest)