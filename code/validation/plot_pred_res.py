from collections import defaultdict


import pandas as pd
import seaborn as sns
import random

import numpy as np



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

import warnings
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import seaborn as sns


all_sr = ['bpd', 'cfs','crohnsdisease', 'dementia',  'depression',\
		'diabetes', 'dysautonomia', 'gastroparesis','hypothyroidism', 'ibs', \
		'interstitialcystitis', 'kidneystones', 'menieres', 'multiplesclerosis',\
		'parkinsons', 'psoriasis', 'rheumatoid', 'sleepapnea']


all_dis = {el:i for i, el in enumerate(all_sr)}
disease_values_dict = all_dis
# these will be used to take disease names for each prediction task
disease_names = list(disease_values_dict.keys())
disease_labels = list(disease_values_dict.values())


def plot_barplot_stats():

	df = constant_data_from_mutliclass_red()

	df['MED-DL'] = df.DL
	df['MetaMap'] = df.MM

	fig, ax = plt.subplots(figsize=(16,8))

	sns.set_style("white")
	sns.set_context("paper")

	df = df.sort_index(ascending=True)
	#df_liwc = df_liwc.sort_index(ascending=True)

	df['MED-DL'] = df.DL
	df['MetaMap'] = df.MM
	df['DIS-LIWC'] = df.DISLIWC

	d = df[['MED-DL', 'MetaMap', 'DIS-LIWC']]

	colors = ["salmon", "y", "lightskyblue"]

	d.plot.bar(ax=ax, yerr=df[['DLe', 'MMe', 'DISLIWCe']].values.T, edgecolor = "white", color=colors, alpha=0.777, ecolor='gray', capsize=3)

	#plt.xticks(rotation=60)

	# for tick in ax.get_xticklabels():
	#     tick.set_rotation(45)

	plt.axhline(y=0.055527352340987664, ls='--', color='red')

	ax.set_xlabel('')

	ax.set_ylim(0,1.15)

	for item in ([ax.xaxis.label, ax.yaxis.label] +
				 ax.get_yticklabels() + ax.get_xticklabels()):
				item.set_fontsize(22)
				item.set_weight('bold')
			

	#labels = ax.yaxis.label
	#print (labels)
	#labels[0] = labels[-1] = ""
	#ax.set_yticklabels(labels)

	labels = [tick.get_text() for tick in ax.get_yticklabels()]
	#ax.set_yticklabels(labels)

	#ax.yaxis.set_major_locator(MaxNLocator(prune='upper'))
			
	l = plt.legend(bbox_to_anchor=(0.01, 1), loc=2, borderaxespad=0., prop={'size':20, 'weight':'bold'})
	l.draw_frame(False)


	plt.tight_layout()

	plt.savefig('results/multiclass/acc_per_sr_DL_MM.png', dpi=100)



def constant_data_from_mutliclass_red():

	#mm = {'bpd': 0.1523095315974573, 'cfs': 0.2085089710050015, 'crohnsdisease': 0.16065089892648315, 'dementia': 0.16173657186541746, 'depression': 0.25150127264912336, 'diabetes': 0.2618807143219669, 'dysautonomia': 0.12515672935841002, 'gastroparesis': 0.23455853407637398, 'hypothyroidism': 0.2152163069171266, 'ibs': 0.212777015112265, 'interstitialcystitis': 0.17490534622610093, 'kidneystones': 0.330755816953, 'menieres': 0.46614447560908534, 'multiplesclerosis': 0.1718457921187137, 'parkinsons': 0.24304973693989712, 'psoriasis': 0.3208271874690978, 'rheumatoid': 0.20177084707745085, 'sleepapnea': 0.26129086329583623}
	#mm_std = {'bpd': 0.03364561837262415, 'cfs': 0.09451324348858432, 'crohnsdisease': 0.064249454834217, 'dementia': 0.03896645063240128, 'depression': 0.07345274531482447, 'diabetes': 0.06886779899673756, 'dysautonomia': 0.061579010092582515, 'gastroparesis': 0.050815950138474335, 'hypothyroidism': 0.07448236189402296, 'ibs': 0.06029580491479956, 'interstitialcystitis': 0.03293695942810976, 'kidneystones': 0.00884704926707396, 'menieres': 0.038286177504798395, 'multiplesclerosis': 0.06961682511273085, 'parkinsons': 0.05787713984068314, 'psoriasis': 0.04778176632063893, 'rheumatoid': 0.0980458495145143, 'sleepapnea': 0.08303759230969858}

	mm = {'bpd': 0.14947778207369, 'cfs': 0.2856198776485409, 'crohnsdisease': 0.16759506953364864, 'dementia': 0.09143049692894895, 'depression': 0.266371021708619, 'diabetes': 0.1887188367260997, 'dysautonomia': 0.10941170323928943, 'gastroparesis': 0.18588806134388186, 'hypothyroidism': 0.17925758196548333, 'ibs': 0.24192570259899973, 'interstitialcystitis': 0.1608071800352439, 'kidneystones': 0.42936706710142525, 'menieres': 0.4175872108509025, 'multiplesclerosis': 0.15398133106102746, 'parkinsons': 0.2552373343191142, 'psoriasis': 0.35402809743190866, 'rheumatoid': 0.16284857715442422, 'sleepapnea': 0.2883909137416431}
	mm_std = {'bpd': 0.08001050239992746, 'cfs': 0.09767475438365528, 'crohnsdisease': 0.09551949048489433, 'dementia': 0.036327941334397665, 'depression': 0.09917233109034541, 'diabetes': 0.06436550324104062, 'dysautonomia': 0.06662772701496326, 'gastroparesis': 0.11929886687190508, 'hypothyroidism': 0.07593029892601348, 'ibs': 0.09053796187491758, 'interstitialcystitis': 0.08165365566115114, 'kidneystones': 0.0800121492844091, 'menieres': 0.08595664991066783, 'multiplesclerosis': 0.08570988556279353, 'parkinsons': 0.06696806579710715, 'psoriasis': 0.08703113729532326, 'rheumatoid': 0.0953475609934879, 'sleepapnea': 0.07853510223115864}


	#dl = {'bpd': 0.5994467630766139, 'cfs': 0.46465065639447667, 'crohnsdisease': 0.42372148497629886, 'dementia': 0.6421464366950053, 'depression': 0.6002086955386743, 'diabetes': 0.6914535598412026, 'dysautonomia': 0.597653963752621, 'gastroparesis': 0.5607740301192207, 'hypothyroidism': 0.6404930288675453, 'ibs': 0.5356697695329329, 'interstitialcystitis': 0.5709589847582747, 'kidneystones': 0.701119696471, 'menieres': 0.7141774126374696, 'multiplesclerosis': 0.5717494507319394, 'parkinsons': 0.6413772867888324, 'psoriasis': 0.5566475694683543, 'rheumatoid': 0.4943208369684031, 'sleepapnea': 0.6980754876136468}
	#dl_std = {'bpd': 0.028674641150773714, 'cfs': 0.07785536322388978, 'crohnsdisease': 0.03441898676764334, 'dementia': 0.06371763995819128, 'depression': 0.04106838228501081, 'diabetes': 0.05597486046786384, 'dysautonomia': 0.03074514063304391, 'gastroparesis': 0.04768117692381654, 'hypothyroidism': 0.04276913765837643, 'ibs': 0.03905708826244867, 'interstitialcystitis': 0.04594899958098184, 'kidneystones': 0.05062041281527265, 'menieres': 0.02260573724849223, 'multiplesclerosis': 0.036644004881119664, 'parkinsons': 0.06672071711298878, 'psoriasis': 0.027368181537754072, 'rheumatoid': 0.02475064179354843, 'sleepapnea': 0.04160032366071243}

	dl = {'bpd': 0.6553978716561668, 'cfs': 0.554593460222974, 'crohnsdisease': 0.4084516826789969, 'dementia': 0.6748442431195305, 'depression': 0.5966281135479627, 'diabetes': 0.6436232614131366, 'dysautonomia': 0.5784548444146357, 'gastroparesis': 0.5549642232697097, 'hypothyroidism': 0.6923015615766892, 'ibs': 0.5308090156922896, 'interstitialcystitis': 0.569766055036607, 'kidneystones': 0.7090994669961483, 'menieres': 0.7168541032380046, 'multiplesclerosis': 0.5751799596824767, 'parkinsons': 0.6426624070240474, 'psoriasis': 0.6493813277464312, 'rheumatoid': 0.5319928899843951, 'sleepapnea': 0.6857176843281992}
	dl_std = {'bpd': 0.06894638085749513, 'cfs': 0.05795346973939952, 'crohnsdisease': 0.06053823813121324, 'dementia': 0.0628927270500837, 'depression': 0.057511851716366434, 'diabetes': 0.043138887174200204, 'dysautonomia': 0.0817007575744207, 'gastroparesis': 0.08985651474819424, 'hypothyroidism': 0.06438860564395245, 'ibs': 0.09638033889212554, 'interstitialcystitis': 0.08144307582722268, 'kidneystones': 0.027079699912537197, 'menieres': 0.0562018513063459, 'multiplesclerosis': 0.0658987707775531, 'parkinsons': 0.06334869575432958, 'psoriasis': 0.062610849868, 'rheumatoid': 0.02475064179354843, 'sleepapnea': 0.04160032366071243}

	liwc = {'bpd': 0.11208406304728548, 'cfs': 0.012738853503184712, 'crohnsdisease': 0.19522776572668113, 'dementia': 0.30833333333333335, 'depression': 0.3835051546391752, 'diabetes': 0.5900621118012421, 'dysautonomia': 0.07111111111111111, 'gastroparesis': 0.07650273224043716, 'hypothyroidism': 0.2222222222222222, 'ibs': 0.035714285714285726, 'interstitialcystitis': 0.0, 'kidneystones': 0.0816326530612245, 'menieres': 0.02395209580838323, 'multiplesclerosis': 0.05376344086021505, 'parkinsons': 0.12060301507537688, 'psoriasis': 0.34854771784232363, 'rheumatoid': 0.0, 'sleepapnea': 0.033898305084745756}
	liwce = {'bpd': 1.3877787807814457e-17, 'cfs': 0.0, 'crohnsdisease': 0.0, 'dementia': 5.551115123125783e-17, 'depression': 5.551115123125783e-17, 'diabetes': 1.1102230246251565e-16, 'dysautonomia': 0.0, 'gastroparesis': 0.0, 'hypothyroidism': 2.7755575615628914e-17, 'ibs': 6.938893903907228e-18, 'interstitialcystitis': 0.0, 'kidneystones': 0.0, 'menieres': 0.0, 'multiplesclerosis': 0.0, 'parkinsons': 1.3877787807814457e-17, 'psoriasis': 0.0, 'rheumatoid': 0, 'sleepapnea': 6.938893903907228e-18}



	all_f1 = {}
	for el in mm:
		all_f1[el] = {}

	for el in mm:
		all_f1[el]['MM'] = mm[el]
		all_f1[el]['MMe'] = mm_std[el]
		all_f1[el]['DL'] = dl[el]
		all_f1[el]['DLe'] = dl_std[el]
		all_f1[el]['DISLIWC'] = liwc[el]
		all_f1[el]['DISLIWCe'] = liwce[el]


	df = pd.DataFrame.from_dict(all_f1, orient='index')
	print (df)
	return df


plot_barplot_stats()