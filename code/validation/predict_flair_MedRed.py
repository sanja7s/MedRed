from flair.data import Corpus
from flair.data import Sentence 

from flair.embeddings import TokenEmbeddings, WordEmbeddings, \
	 StackedEmbeddings, CharacterEmbeddings, FlairEmbeddings, \
	 PooledFlairEmbeddings, ELMoEmbeddings, BertEmbeddings , RoBERTaEmbeddings
from typing import List

from create_flair_corpus import read_in_AMT, read_in_CADEC

# 6. initialize trainer
from flair.trainers import ModelTrainer

# 5. initialize sequence tagger
from flair.models import SequenceTagger

# 6. initialize trainer
from flair.training_utils import EvaluationMetric

# 9. continue trainer at later point
from pathlib import Path

import pandas as pd
import tqdm


f_in = 'data/AMT/pred/all_Reddit_AMT_labels.csv'
f_out = 'results/NER_res/AMT/NER_res_AMT_sent.csv'

def predict(model, selected_embeddings, data_file):

	"""
			takes data in a form text, post_id, and saves both those plus 
			prediction results in the out file
	"""

	selected_embeddings_text = [key  for key in selected_embeddings if selected_embeddings[key]]
	selected_embeddings_text = '_'.join(selected_embeddings_text)

	print (selected_embeddings_text)

	model_dir = 'resources/taggers/' + model + selected_embeddings_text + '_fine-tuned7s'

	# load the model you trained
	model = SequenceTagger.load(model_dir + '/best-model.pt')

	data = pd.read_csv(f_in)
	# ,year,month,subreddit,body,clean_body,post_index
	print(data.head())

	with open(f_out.replace(".csv", "_drug.csv"), 'w') as f_drug:
		with open(f_out.replace(".csv", "_dis.csv"), 'w') as f_dis:
			header = "subreddit,post_index,matched,score,start_pos,end_pos\n"
			f_dis.write(header)
			f_drug.write(header)

			for i, row in tqdm.tqdm(data.iterrows(), total=data.shape[0]):
				sentence = Sentence(str(row['post']))
				# print (r)
				# # predict tags and print
				model.predict(sentence)
				res = sentence.to_dict(tag_type='ner')

				for el in res['entities']:
					
					if el['type'] == 'DIS':
						f_dis.write(row['subreddit']+','+row['post_index']+',"'+\
							el['text'].replace('\n', ' ').replace('\t', ' ')+'",'+str(el['confidence'])+','+str(el['start_pos'])+','+str(el['end_pos'])+'\n')
					elif el['type'] == 'DRUG':
						f_drug.write(row['subreddit']+','+row['post_index']+',"'+\
							el['text'].replace('\n', ' ')+'",'+str(el['confidence'])+','+str(el['start_pos'])+','+str(el['end_pos'])+'\n')
					

				# if i ==10:
				# 	break


model = 'AMT'
selected_embeddings = {'glove':1, 'char':0, 'flair':0, 'pooled-flair':0, \
						'bert':0, 'twitter':0, 'elmo':0, 'roberta':1}
predict(model, selected_embeddings, f_in)