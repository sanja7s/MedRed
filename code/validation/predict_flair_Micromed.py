from flair.data import Corpus
from flair.data import Sentence 

from flair.embeddings import TokenEmbeddings, WordEmbeddings, \
	 StackedEmbeddings, CharacterEmbeddings, FlairEmbeddings, \
	 PooledFlairEmbeddings, ELMoEmbeddings, BertEmbeddings , RoBERTaEmbeddings
from typing import List

from create_flair_corpus import read_in_Micromed

# 6. initialize trainer
from flair.trainers import ModelTrainer

# 5. initialize sequence tagger
from flair.models import SequenceTagger

# 6. initialize trainer
from flair.training_utils import EvaluationMetric

# 9. continue trainer at later point
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import bz2, re
import mmap

import preprocessor as p

p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.SMILEY)


# we cannot provide these data but they can be downloaded from the original publication
f_in = 'data/twitter_data/sample.bz'         # 
f_out = 'results/NER_res/twitter/entities.csv'

def get_num_lines(file_path):
	with bz2.open(file_path) as fh:
		lines = 0
		for _ in fh:
			lines += 1
	return lines


def get_clean_body(body):
	return re.findall(r"[\w']+|[/().,!?;]", body)

def preprocess_body(body):
	return p.clean(body)

def process_txt(line):
	cur_dict = line.split('\t')

	whole_dict ={}

	whole_dict['id'] = 'tid' + cur_dict[2]
	whole_dict['created_at'] = cur_dict[1]
	whole_dict['text'] = cur_dict[3]
	whole_dict['user_name'] = cur_dict[0]
	whole_dict['user_loc'] = cur_dict[4] + ', ' + cur_dict[5].strip()
	print (cur_dict[3])
	clean_tweet = preprocess_body(cur_dict[3])
	print (clean_tweet)
	whole_dict['clean_body'] = clean_tweet

	return whole_dict


def predict(model, selected_embeddings, data_file):

	"""
			takes data in a form text, post_id, and saves both those plus 
			prediction results in the out file
	"""

	selected_embeddings_text = [key  for key in selected_embeddings if selected_embeddings[key]]
	selected_embeddings_text = '_'.join(selected_embeddings_text)

	print (selected_embeddings_text)


	model_dir = 'resources/taggers/CADECglove_char_flair'            # 

	# load the model you trained
	model = SequenceTagger.load(model_dir + '/best-model.pt')

	line_counts = 0

	with bz2.open(f_in, 'rt') as f:

		with open(f_out.replace(".csv", "_drug.csv"), 'w') as f_drug:
			with open(f_out.replace(".csv", "_dis.csv"), 'w') as f_dis:

				header = "post_ID,matched,score,start_pos,end_pos\n"
				f_dis.write(header)
				f_drug.write(header)

			
				for line in tqdm(f, total=get_num_lines(f_in)):
					if len(line) > 0:
						line_dict = process_txt(line)
						line_counts += 1

						body = line_dict['text']
						tweet_id = line_dict['id']


						sentence = Sentence(str(body))
						# print (r)
						# # predict tags and print

						model.predict(sentence)
						res = sentence.to_dict(tag_type='ner')

						for el in res['entities']:

							if el['type'] == 'DIS':
								f_dis.write(tweet_id+',"'+\
									el['text'].replace('\n', ' ')+'",'+str(el['confidence'])+','+str(el['start_pos'])+','+str(el['end_pos'])+'\n')
							elif el['type'] == 'DRUG':
								f_drug.write(tweet_id+',"'+\
									el['text'].replace('\n', ' ')+'",'+str(el['confidence'])+','+str(el['start_pos'])+','+str(el['end_pos'])+'\n')
					

					# if line_counts ==50:
					# 	break


model = 'CADEC'
selected_embeddings = {'glove':1, 'char':0, 'flair':0, 'pooled-flair':0, \
						'bert':0, 'twitter':0, 'elmo':0, 'roberta':1, \
						'biobert':0, 'clinicalbiobert':0}
predict(model, selected_embeddings, f_in)