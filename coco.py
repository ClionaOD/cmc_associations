import pandas as pd

coco = pd.read_csv('./coco_classes.txt',sep=':',header=None)
coco.columns = ['index','labels']
things = coco['labels'].to_list()

coco_stuff = pd.read_csv('./coco_stuff_labels.txt',sep=':',header=None)
coco_stuff.columns = ['index','labels']


for idx in range(len(coco_stuff)):
    label = coco_stuff.loc[idx]['labels']
    if label in things:
        coco_stuff.loc[idx,'type'] = 'thing'
    elif label == 'unlabeled':
        coco_stuff.loc[idx,'type'] = label
    else:
        coco_stuff.loc[idx,'type'] = 'stuff'

coco_stuff.to_csv('./cocostuff_labels.csv')