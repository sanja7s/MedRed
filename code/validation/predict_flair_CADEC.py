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


f_in = 'data/CADEC/pred/cadec.csv'
f_out = 'results/NER_res/CADEC/to_resume_NER_res_CADEC_sent.csv'

def predict(model, selected_embeddings, data_file):

	"""
			takes data in a form text, post_id, and saves both those plus 
			prediction results in the out file
	"""

	selected_embeddings_text = [key  for key in selected_embeddings if selected_embeddings[key]]
	selected_embeddings_text = '_'.join(selected_embeddings_text)

	print (selected_embeddings_text)

	model_dir = 'resources/taggers/' + 'to_resume_' + model + selected_embeddings_text

	# load the model you trained
	model = SequenceTagger.load(model_dir + '/best-model.pt')

	data = pd.read_csv(f_in)
	# ,year,month,subreddit,body,clean_body,post_index
	print(data.head())

	with open(f_out.replace(".csv", "_drug.csv"), 'w') as f_drug:
		with open(f_out.replace(".csv", "_dis.csv"), 'w') as f_dis:
			header = "post_ID,matched,score,start_pos,end_pos\n"
			f_dis.write(header)
			f_drug.write(header)

			for i, row in tqdm.tqdm(data.iterrows(), total=data.shape[0]):
				#r = ' '.join(eval(row['body']))
				for r in eval(row['body']):
					sentence = Sentence(str(r))
					# print (r)
					# # predict tags and print
					model.predict(sentence)
					res = sentence.to_dict(tag_type='ner')

					for el in res['entities']:

						if el['type'] == 'DIS':
							f_dis.write(row['post_ID']+',"'+\
								el['text'].replace('\n', ' ')+'",'+str(el['confidence'])+','+str(el['start_pos'])+','+str(el['end_pos'])+'\n')
						elif el['type'] == 'DRUG':
							f_drug.write(row['post_ID']+',"'+\
								el['text'].replace('\n', ' ')+'",'+str(el['confidence'])+','+str(el['start_pos'])+','+str(el['end_pos'])+'\n')
						

				if i ==10:
					break


model = 'CADEC'
selected_embeddings = {'glove':1, 'char':0, 'flair':0, 'pooled-flair':0, \
						'bert':0, 'twitter':0, 'elmo':0, 'roberta':1}
predict(model, selected_embeddings, f_in)