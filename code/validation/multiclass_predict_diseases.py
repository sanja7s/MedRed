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

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams.update({'font.size': 16})


features_file = "data/features/{}_embdedded_features.pckl".format(etype)
results_file = "results/{}_multiclasscm.csv".format(etype)

word_emb_len = 300


def sample_all_diseases(df, n=1):

    if etype == "DL":
        smallest_disease=all_dis['parkinsons']
    else:
        smallest_disease=all_dis['gastroparesis']
    
    def merge_rows(row):
        if n == 1:
            return row
        res_row = np.zeros(len(row[0]))
        for i in range(n):
                res_row = res_row+row[i]
        return res_row / n
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    dis_size = len(df[df['disease']==smallest_disease])
    sample_size = int(dis_size/n)*n
    
    print(dis_size, sample_size)
    
    df_sample= pd.DataFrame()
    for disease in all_dis:
        df_dis = df[df['disease'] == all_dis[disease]]
        df_dis = df_dis.sample(n=sample_size, random_state=11).reset_index()
        if n > 1:
            df_dis = df_dis.groupby(df_dis.index // n).agg(lambda x: list(x))
        df_dis['disease'] = all_dis[disease]
        df_sample = pd.concat([df_dis, df_sample])
    
    
    if n > 1:
        df_sample['features'] =  df_sample['features'].apply(lambda row: merge_rows(row))
    df_sample = df_sample.drop(columns=['index'])
    
    return df_sample



def prepare_training_data_for_multi_disease(features, n=1):
    
    dis_sample = sample_all_diseases(features, n)
    print("Subsampled all diseases for ", len(dis_sample), " posts")
    
    training = dis_sample.copy()
    training = training.reset_index(drop=True)

    return training


def XGBoost_cross_validate():

    features = pd.read_pickle(features_file)

    features.rename(columns={'vec':'features'}, inplace=True)
    features = features.drop(columns=['subreddit', 'entities'])

    disease = features['disease']
    print ("Post per subreddit ")
    print (features.groupby('disease').size())

    # print('Distribution before imbalancing: {}'.format(Counter(disease)))

    training = prepare_training_data_for_multi_disease(features)
    print(training.tail())
    
    training_labels = training["disease"].astype(int)
    training_labels.head()

    training_features = pd.DataFrame(training["features"].tolist())
    training_features.head()
    
    # XGBoost
    AUC_results = []
    f1_results = []
    results = []

    cm_all = []


    kf = StratifiedKFold(n_splits=10, random_state=11, shuffle=True)

    for train_index, test_index in kf.split(training_features,training_labels):
        X_train = training_features.loc[train_index]
        y_train = training_labels.loc[train_index]

        X_test = training_features.loc[test_index]
        y_test = training_labels.loc[test_index]


        model = XGBClassifier(n_estimators=100, n_jobs=11, max_depth=4)  # 1000 200
        model.fit(X_train, y_train.values.ravel())
        predictions = model.predict(X_test)

        results.append(precision_recall_fscore_support(y_test, predictions))
        f1_results.append(f1_score(y_true=y_test, y_pred=predictions, average='weighted'))

        cm_cv = sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=predictions, labels=disease_labels)
        cm_all.append(cm_cv)

        print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, predictions))


    f1_results_avg = [pd.np.mean(f1_results), pd.np.std(f1_results)]
    #AUC_results_avg = [pd.np.mean(AUC_results), pd.np.std(AUC_results)]

    print (f1_results_avg)
    
    return f1_results, results, model, cm_all


def plot_confusion_matrix():

    f1_results, results, model, cm_all = XGBoost_cross_validate()

    results_avg = pd.np.mean(results, axis=0)
    f1 = results_avg[2]
    per_dis_f1 = [ str(disease_names[i]) + ' F1: ' + "{0:.2f}".format(f1[i]) for i in range (len(f1)) ]

    cms = np.array(cm_all)
    cms2 = cms.sum(axis=0)

    from matplotlib.colors import LogNorm 
    from matplotlib import cm
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

    sns.set_style('darkgrid')

    syn = 'royalblue'
    sem = 'darkorange'
    join = 'forestgreen'

    # normalize confusion matrix
    #cms2 = np.round(cms2.astype('float') / cms2.sum(axis=1)[:, np.newaxis],2)


    viridis = cm.get_cmap('viridis', 12)
    a = sns.heatmap(cms2, square=True, cbar=0,
            #normalize=True,
            #norm=LogNorm(vmin=cms2.min(), vmax=cms2.max()),
            cmap=viridis,
            xticklabels=disease_names,
            yticklabels=per_dis_f1, annot=True, fmt='1g', ax=ax, annot_kws={"size": 13, "weight": "bold"})
    #     a.xaxis.tick_top()
    #     a.title.
    #     a.xaxis.
    #ax.set_title(i)
        

    plt.tight_layout()

    fig.savefig('results/multiclass/classifier_for_' + etype  + '_cm_bold_v4.png')


    results_std = pd.np.std(results, axis=0)
    f1_std = results_std[2]


    per_dis_f1_dict = {str(disease_names[i]): f1[i] for i in range (len(f1)) }
    per_dis_f1_dict_std = {str(disease_names[i]): f1_std[i] for i in range (len(f1)) }

    print (per_dis_f1_dict)
    print(per_dis_f1_dict_std)


def save_confusion_matrix():


    f1_results, results, model, cm_all = XGBoost_cross_validate()

    cms = np.array(cm_all)
    cms2 = cms.sum(axis=0)

    np.savetxt(results_file, cms2)


plot_confusion_matrix()