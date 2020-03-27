import spacy
# from collections import defaultdict
nlp = spacy.load('en_core_web_lg')

import pandas as pd
import sys
# import random
import pickle
import numpy as np

import ast

import time
start_time = time.time()

all_sr = ['bpd', 'cfs','crohnsdisease', 'dementia',  'depression',\
                    'diabetes', 'dysautonomia', 'gastroparesis','hypothyroidism', 'ibs', \
                    'interstitialcystitis', 'kidneystones', 'menieres', 'multiplesclerosis',\
                    'parkinsons', 'psoriasis', 'rheumatoid', 'sleepapnea']



etype = "DL"
sample = "_sample"

sym_file = "data/entities/{}/{}_symptom_mappings.csv".format(etype, etype)
drug_file = "data/entities/{}/{}_drugs_mappings.csv".format(etype, etype)
features_file = "data/features/{}_embdedded_features{}.pckl".format(etype, sample)

sym = pd.read_csv(sym_file) # ,subreddit,matched,UID,norm_UID,post_index,score
drug = pd.read_csv(drug_file) # ,post_index,subreddit,matched,norm_UID,UID,score

if sample == "_sample":
    data = sym.append(drug).sample(n=1000, random_state=7)
else:
    data = sym.append(drug)
print ("Total entities ", len(data))

data = data[data['subreddit'].isin(all_sr)]
print ("Total entities ", len(data))



word_emb_len = 300
def get_embedding_vec(tokens):
    """
        find average embedding
        for all the tokens in
        the symptoms list
    """
    vec = []
    vec.append(np.zeros(word_emb_len))
    for token in tokens:
        if token.has_vector:
            vec.append(token.vector)
            #print(vec)
    return np.mean(vec, axis=0)

def embedding_from_tokens(row):
    tokens = nlp(row)
    vec = get_embedding_vec(tokens)
    return vec.tolist()

def save_features_with_certainty(certainty, features_file=features_file):

    # Clean
    raw_features = data[["subreddit", "matched", "post_index", "score"]]

    print (len(raw_features))

    raw_features = raw_features[ (raw_features["score"].astype(float) > certainty) ]

    print (len(raw_features))

    if not len(raw_features):
        return

    if etype == "DL":
        #raw_features['matched'] = raw_features['matched'].apply(','.join)
        pass
    elif etype == "MM":
        raw_features['matched'] = raw_features['matched'].apply(ast.literal_eval)
        raw_features['matched'] = raw_features['matched'].apply(' '.join)
    else:
        print ("Non-existent entitiy type, please try again. ")
        sys.exit()

    raw_features = raw_features.rename(columns={'matched':'entities'})

    raw_features = raw_features.groupby(['post_index','subreddit'])['entities'].apply(', '.join).reset_index()
    raw_features = raw_features.drop(columns=['post_index'])
    print ("Total posts with entities ", len(raw_features))


    features_file = features_file.replace(".pckl", "_{:.2f}.pckl".format(certainty))

    object_features = raw_features.astype(object) # in order to add vectors to cells
    object_features.head()

    object_features['vec'] = object_features['entities'].apply(embedding_from_tokens)


    embedding_vec_list = object_features['vec'].tolist()
    embedding_vec_list = pd.DataFrame(embedding_vec_list)


    features = object_features.copy()

    all_dis = {el:i for i, el in enumerate(all_sr)}
    disease_values_dict = all_dis
    disease_values_dict

    # these will be used to take disease names for each prediction task
    disease_names = list(disease_values_dict.keys())
    disease_labels = list(disease_values_dict.values())


    s = pd.DataFrame()
    s['disease'] = features.apply(lambda x: disease_values_dict[x['subreddit']], axis=1)

    features = features.join(s)
    features.to_pickle(features_file)


def save_features_given_num_entities_per_post(ne, maxne, features_file=features_file):

    # Clean
    raw_features = data[["subreddit", "matched", "post_index", "score"]]

    # print (len(raw_features))
    if not len(raw_features):
        return

    if etype == "DL":
        #raw_features['matched'] = raw_features['matched'].apply(','.join)
        pass
    elif etype == "MM":
        raw_features['matched'] = raw_features['matched'].apply(ast.literal_eval)
        raw_features['matched'] = raw_features['matched'].apply(' '.join)
    else:
        print ("Non-existent entitiy type, please try again. ")
        sys.exit()

    raw_features = raw_features.rename(columns={'matched':'entities'})

    raw_features = raw_features.groupby(['post_index','subreddit'])['entities'].apply(list).reset_index() 


    if ne != maxne:
        raw_features = raw_features[ (raw_features['entities'].apply(len) == ne) ]
    else:
        raw_features = raw_features[ (raw_features['entities'].apply(len) >= ne) ]
        

    print (raw_features["entities"].head(2))
    print ("With {} entities per post we got {} posts.".format(ne, len(raw_features)))
    if not len(raw_features):
        return

    raw_features["entities"] = raw_features["entities"].apply(', '.join)

    raw_features = raw_features.drop(columns=['post_index'])
    print ("Total posts with entities ", len(raw_features))


    features_file = features_file.replace(".pckl", "_ent_per_post_{}.pckl".format(ne))

    object_features = raw_features.astype(object) # in order to add vectors to cells
    object_features.head()

    object_features['vec'] = object_features['entities'].apply(embedding_from_tokens)


    embedding_vec_list = object_features['vec'].tolist()
    embedding_vec_list = pd.DataFrame(embedding_vec_list)


    features = object_features.copy()

    all_dis = {el:i for i, el in enumerate(all_sr)}
    disease_values_dict = all_dis
    disease_values_dict

    # these will be used to take disease names for each prediction task
    disease_names = list(disease_values_dict.keys())
    disease_labels = list(disease_values_dict.values())


    s = pd.DataFrame()
    s['disease'] = features.apply(lambda x: disease_values_dict[x['subreddit']], axis=1)

    features = features.join(s)
    features.to_pickle(features_file)



def produce_certainty_fa_features():
    for certainty in np.linspace(0.1,1,9, endpoint=False):
        save_features_with_certainty(certainty)


def produce_ne_fa_features():
    maxne = 8
    for ne in range(maxne,maxne+1):
        save_features_given_num_entities_per_post(ne,maxne)






print("--- %s seconds ---" % (time.time() - start_time))














