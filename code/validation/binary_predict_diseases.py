# import spacy
from collections import defaultdict
# nlp = spacy.load('en_core_web_lg')

import pandas as pd
import seaborn as sns
import random
import pickle
import numpy as np

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
from collections import Counter

import sklearn
#from sklearn.pipeline import Pipeline
from sklearn import linear_model
#from sklearn import svm
#from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

from sklearn.model_selection import KFold #cross_validate, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


all_sr = ['bpd', 'cfs','crohnsdisease', 'dementia',  'depression',\
                    'diabetes', 'dysautonomia', 'gastroparesis','hypothyroidism', 'ibs', \
                    'interstitialcystitis', 'kidneystones', 'menieres', 'multiplesclerosis',\
                    'parkinsons', 'psoriasis', 'rheumatoid', 'sleepapnea']


all_dis = {el:i for i, el in enumerate(all_sr)}
disease_values_dict = all_dis
# these will be used to take disease names for each prediction task
disease_names = list(disease_values_dict.keys())
disease_labels = list(disease_values_dict.values())


etype="DL"


features_file = "data/features/{}_embdedded_features.pckl".format(etype)
results_file = "results/{}_all_res_n1.csv".format(etype)

word_emb_len = 300




def sample_one_disease(df, disease, n):
    
    def merge_rows(row):
        if n == 1:
            return row
        res_row = np.zeros(len(row[0]))
        for i in range(n):
                res_row = res_row+row[i]
        return res_row / n
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    dis_size = len(df[df['disease']==disease])
    sample_size = int(dis_size/n)*n
    #
    print(dis_size, sample_size)
    
    df_dis = df[df['disease'] == disease]
    df_dis = df_dis.sample(n=sample_size, random_state=7).reset_index()
    if n > 1:
        df_dis = df_dis.groupby(df_dis.index // n).agg(lambda x: list(x))
    df_dis['disease'] = 1
    
    
    df_others = df[df['disease'] != disease]
    df_others = df_others.sample(n=sample_size, random_state=7).reset_index()
    if n > 1:
        df_others = df_others.groupby(df_others.index // n).agg(lambda x: list(x))
    df_others['disease'] = 0
    
    
    df_sample = pd.concat([df_dis, df_others]) #.sample(frac=1)
    if n > 1:
        df_sample['features'] =  df_sample['features'].apply(lambda row: merge_rows(row))
    df_sample = df_sample.drop(columns=['index'])
    
    return df_sample




def prepare_training_data_for_one_disease(DISEASE7s, features, n):
    
    disease_names_labels = ['others', disease_names[DISEASE7s]]
    
    dis_sample = sample_one_disease(features, DISEASE7s, n)
    print("Subsampled ", disease_names[DISEASE7s], "for ", len(dis_sample), " posts")
    
    training = dis_sample.copy()
    training = training.reset_index(drop=True)

    return training




def XGBoost_cross_validate(training, disease_number_labels):
    
    training_labels = training["disease"].astype(int)
    training_labels.head()

    training_features = pd.DataFrame(training["features"].tolist())
    training_features.head()
    
    # XGBoost
    AUC_results = []
    f1_results = []
    results = []

    cm_all = []

    kf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)

    for train_index, test_index in kf.split(training_features,training_labels):
        X_train = training_features.loc[train_index]
        y_train = training_labels.loc[train_index]

        X_test = training_features.loc[test_index]
        y_test = training_labels.loc[test_index]


        model = XGBClassifier(n_estimators=1000, n_jobs=11, max_depth=4)  # 1000 200
        model.fit(X_train, y_train.values.ravel())
        predictions = model.predict(X_test)

        results.append(precision_recall_fscore_support(y_test, predictions))
        f1_results.append(f1_score(y_true=y_test, y_pred=predictions, average='weighted'))
        AUC_results.append(metrics.roc_auc_score(y_test, predictions))

        cm_cv = sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=predictions, labels=disease_number_labels)
        cm_all.append(cm_cv)

        #print ("AUC Score : %f" % metrics.roc_auc_score(y_test, predictions))
        #print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, predictions))


    f1_results_avg = [pd.np.mean(f1_results), pd.np.std(f1_results)]
    AUC_results_avg = [pd.np.mean(AUC_results), pd.np.std(AUC_results)]
    
    return f1_results_avg, AUC_results_avg, results, model

def XGBoost_cross_validate_ne(training, disease_number_labels):
    
    training_labels = training["disease"].astype(int)
    training_labels.head()

    training_features = pd.DataFrame(training["features"].tolist())
    training_features.head()
    
    # XGBoost
    AUC_results = []
    f1_results = []
    results = []

    cm_all = []

    kf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)

    for train_index, test_index in kf.split(training_features,training_labels):
        X_train = training_features.loc[train_index]
        y_train = training_labels.loc[train_index]

        X_test = training_features.loc[test_index]
        y_test = training_labels.loc[test_index]


        model = XGBClassifier(n_estimators=1000, n_jobs=11, max_depth=4)  # 1000 200
        model.fit(X_train, y_train.values.ravel())
        predictions = model.predict(X_test)

        results.append(precision_recall_fscore_support(y_test, predictions))
        f1_results.append(f1_score(y_true=y_test, y_pred=predictions, average='weighted'))
        AUC_results.append(metrics.roc_auc_score(y_test, predictions))

        cm_cv = sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=predictions, labels=disease_number_labels)
        cm_all.append(cm_cv)

        #print ("AUC Score : %f" % metrics.roc_auc_score(y_test, predictions))
        #print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, predictions))


    # f1_results_avg = [pd.np.mean(f1_results), pd.np.std(f1_results)]
    # AUC_results_avg = [pd.np.mean(AUC_results), pd.np.std(AUC_results)]
    
    return f1_results, results, model


def eval_functions(f1_results_avg, AUC_results_avg, results):
    results_avg = pd.np.mean(results, axis=0)
    results_std = pd.np.std(results, axis=0)
    P_res = np.array([np.mean(results_avg[0]), np.mean(results_std[0])])
    R_res = np.array([np.mean(results_avg[1]), np.mean(results_std[1])])
    support_res = np.array([np.mean(results_avg[3]), np.mean(results_std[3])])
    print("F1 average score ", f1_results_avg, np.mean(results_avg[2]))
    print("AUC average score ", AUC_results_avg)
    print("P average score ", P_res)
    print("R average score ", R_res)
    print("Support per class ", support_res)
    
    return {'F1':f1_results_avg, 'AUC': AUC_results_avg, 'support': support_res,
           'P': P_res, 'R': R_res}


def predict_with_certainty(certainty, features_file=features_file, results_file=results_file):

    features_file = features_file.replace(".pckl", "_{:.2f}.pckl".format(certainty))
    results_file = results_file.replace(".csv", "_{:.2f}.csv".format(certainty))


    features = pd.read_pickle(features_file)


    features.rename(columns={'vec':'features'}, inplace=True)
    features = features.drop(columns=['subreddit', 'entities'])

    disease = features['disease']
    print ("Post per subreddit ")
    print (features.groupby('disease').size())



    # get the classes sizes
    min_class_size = min(features.groupby('disease').size())
    min_class_size

    print('Distribution before imbalancing: {}'.format(Counter(disease)))


    random_state7s = 3

    #disease_number_labels = [0, 1]
    #disease_names_labels = ['others', disease_names[DISEASE7s]]

    all_res = defaultdict(int)
    n = 1
    for DISEASE7s in disease_labels:
        
        disease_number_labels = [0, 1]
        disease_names_labels = ['others', disease_names[DISEASE7s]]
        
        balanced_features = prepare_training_data_for_one_disease(DISEASE7s, features, n)
        balanced_features.tail()
        
        f1_results_avg, AUC_results_avg, results, model = \
            XGBoost_cross_validate(balanced_features, disease_number_labels)
        
        print("RESULTS for ~~~~~~~~~~~~~~~~ ", disease_names[DISEASE7s], str(DISEASE7s), "~~~~~~~~~~~~~~~~~")
        res = eval_functions(f1_results_avg, AUC_results_avg, results)
        all_res[disease_names[DISEASE7s]] =  res
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        
        # pickle.dump(model, open("results/models/" + 
        #     disease_names[DISEASE7s] + str(n) + etype + ".pickle.dat", "wb"))


    df_res = pd.DataFrame(all_res)
    df_res.to_csv(results_file)


def predict_with_num_entities_per_post(ne, features_file=features_file, results_file=results_file):

    features_file = features_file.replace(".pckl", "_ent_per_post_{}.pckl".format(ne))
    results_file = results_file.replace(".csv", "_ent_per_post_{}_44.csv".format(ne))

    features = pd.read_pickle(features_file)

    features.rename(columns={'vec':'features'}, inplace=True)
    features = features.drop(columns=['subreddit', 'entities'])

    disease = features['disease']
    print ("Post per subreddit ")
    print (features.groupby('disease').size())


    # get the classes sizes
    min_class_size = min(features.groupby('disease').size())
    min_class_size

    print('Distribution before imbalancing: {}'.format(Counter(disease)))

    random_state7s = 3

    all_res = defaultdict(int)
    n = 1
    for DISEASE7s in disease_labels:
        
        disease_number_labels = [0, 1]
        disease_names_labels = ['others', disease_names[DISEASE7s]]
        
        balanced_features = prepare_training_data_for_one_disease(DISEASE7s, features, n)
        balanced_features.tail()
        
        f1_results, results, model = \
            XGBoost_cross_validate_ne(balanced_features, disease_number_labels)
        
        print("RESULTS for ~~~~~~~~~~~~~~~~ ", disease_names[DISEASE7s], str(DISEASE7s), "~~~~~~~~~~~~~~~~~")
        all_res[disease_names[DISEASE7s]] =  f1_results
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        
        # pickle.dump(model, open("results/models/" + 
        #     disease_names[DISEASE7s] + str(n) + etype + ".pickle.dat", "wb"))


    df_res = pd.DataFrame(all_res)
    df_res.to_csv(results_file)

def extract_support_for_num_entities_per_post(ne, features_file=features_file, results_file=results_file):

    results_file = results_file.replace(".csv", "_support_for_ent_per_post_{}.csv".format(ne))
    features_file = features_file.replace(".pckl", "_ent_per_post_{}.pckl".format(ne))
    
    features = pd.read_pickle(features_file)

    features.rename(columns={'vec':'features'}, inplace=True)
    features = features.drop(columns=['subreddit', 'entities'])

    disease = features['disease']
    print ("Post per subreddit ")
    print (features.groupby('disease').size())


    # get the classes sizes
    min_class_size = min(features.groupby('disease').size())
    min_class_size

    print('Distribution before imbalancing: {}'.format(Counter(disease)))

    random_state7s = 3

    all_res = defaultdict(int)
    n = 1
    for DISEASE7s in disease_labels:
        
        disease_number_labels = [0, 1]
        disease_names_labels = ['others', disease_names[DISEASE7s]]
        
        balanced_features = prepare_training_data_for_one_disease(DISEASE7s, features, n)
        balanced_features.tail()
        
      
        all_res[disease_names[DISEASE7s]] =  len(balanced_features)


    df_res = pd.DataFrame(all_res, index=[0])
    df_res.to_csv(results_file)

# for certainty in np.linspace(0.1,1,9, endpoint=False):
#     predict_with_certainty(certainty)


maxne = 7
for ne in range(6,maxne+1):
    predict_with_num_entities_per_post(ne)


# maxne = 8
# for ne in range(1,maxne+1):
#     extract_support_for_num_entities_per_post(ne)

