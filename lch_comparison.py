import os
import json
import nltk
import pickle 
import argparse
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
#from skbio.stats.distance import mantel

def parse_option():
    parser = argparse.ArgumentParser('arguments for lch calculations')

    parser.add_argument('--image_path', type=str, help='path to images tested, for getting list of classes')
    parser.add_argument('--save_path', type=str, help='path to save correlation results')
    parser.add_argument('--rdm_path', type=str, help='where to find the activation rdms')
    parser.add_argument('--save_lch', type=bool, default=False, help='whether to save the lch_df')
    parser.add_argument('--open_lch', type=bool, default=False, help='whether to open a previously saved lch_df and path')
    parser.add_argument('--lch_path', type=str, help='path to save the lch_df')
    parser.add_argument('--correlation', type=str, default='kendalltau', choices=['kendalltau','spearman','pearson'], help='what correlation value to use')

    opt = parser.parse_args()

    return opt

def calculate_lch(args):
    wnids = os.listdir(args.image_path)
    
    synsets = []
    for wnid in wnids:
        pos = wnid[0]
        offset = int(wnid[1:])
        synsets.append(wn.synset_from_pos_and_offset(pos,offset))
    
    lch_df = pd.DataFrame(
        columns=[_name.name() for _name in synsets], 
        index=[_name.name() for _name in synsets]
    )

    print('calculating lch matrix ...')
    for synset1 in synsets:
        for synset2 in synsets:
            lch_df.loc[synset1.name(), synset2.name()] = synset1.lch_similarity(synset2)
    print('done')

    if args.save_lch:
        with open(f'{args.lch_path}/{len(wnids)}_categs_lch.pickle', 'wb') as f:
            pickle.dump(lch_df,f)
    
    return lch_df

def main(args):
    wnids = os.listdir(args.image_path)
    
    if args.open_lch:
        with open(f'{args.lch_path}/{len(wnids)}_categs_lch.pickle', 'rb') as f:
            lch_df = pickle.load(f)
    else:
        lch_df = calculate_lch(args)

    lch_vals = lch_df.values
    lch_vals = -(lch_vals - lch_vals[0][0])
    np.fill_diagonal(lch_vals,0)
    lch_df = pd.DataFrame(data=lch_vals, index=lch_df.index, columns=lch_df.columns)
    
    corr_results = {k.split('_')[0] : None for k in os.listdir(args.rdm_path)}
    
    for model_file in os.listdir(args.rdm_path):
        if not os.path.isdir(f'{args.rdm_path}/{model_file}'):
            with open(f'{args.rdm_path}/{model_file}', 'rb') as f:
                rdms = pickle.load(f)
            
            model = model_file.split('_')[0]

            results = pd.DataFrame(
                index=list(rdms.keys()),
                columns=[f'{args.correlation} correlation', 'pval']
            )

            for layer,rdm in rdms.items():
                corr, pval, n = mantel(rdm.values, lch_df.values, method=args.correlation)
                results.loc[layer,f'{args.correlation} correlation'] = corr
                results.loc[layer,'pval'] = pval

            corr_results[model] = results

    with open(args.save_path, 'wb') as f:
        pickle.dump(corr_results, f)
        
if __name__ == "__main__":
    args = parse_option()

    #### working code for testing ####
    with open('img_labels.pickle','rb') as f:
        labels = pickle.load(f)

    categs = pd.read_csv('./coco_labels.csv', header=None)

    synonyms= {
        'stop_sign':'signpost',
        'eye_glasses':'spectacles',
        'sports_ball':'ball',
        'wine_glass':'wineglass',
        'potted_plant':'flowerpot',
        'hair_brush':'brush'
    }

    defs = []
    synset = []
    for word in categs[0].to_list():
        if ' ' in word:
            word=word.replace(' ','_')
        syns = wn.synsets(word)
        if not len(syns) == 0:
            defs.append(syns[0].definition())
            synset.append(syns[0])
        else:
            word = synonyms[word]
            defs.append(wn.synsets(word)[0].definition())
            synset.append(wn.synsets(word)[0])
            print(word)
    categs['defs'] = defs
    categs['syns'] = synset

    categs.to_csv('./wn_defs.csv')
    all = []
    for img, lst in labels.items():
        syns = [wn.synsets(word) for word in lst]
        syns = [[j for j in i if j.pos()=='n'] for i in syns]
        all.append(syns)

    unique = [[item for sublst in sub for item in sublst] for sub in all]
    unique = list(set([item for sublst in unique for item in sublst]))
    non_one = [syn for syn in unique if not str(syn)[-3]=='1']
    for syn in non_one:
        print(f"{syn}: \t {syn.definition()}")
    print(unique)

    ########
    main(args) 