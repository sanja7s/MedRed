from flair.data import Corpus
from flair.data import Sentence 

from flair.embeddings import TokenEmbeddings, WordEmbeddings, \
	 StackedEmbeddings, CharacterEmbeddings, FlairEmbeddings, \
	 PooledFlairEmbeddings, ELMoEmbeddings, BertEmbeddings , RoBERTaEmbeddings
from typing import List

from create_flair_corpus import read_in_CADEC

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


def eval(model, selected_embeddings, corpus):

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


	result, _ = model.evaluate(corpus.test)
	print(result.detailed_results)



corpus = read_in_CADEC()

model = 'CADEC'
selected_embeddings = {'glove':1, 'char':0, 'flair':0, 'pooled-flair':0, \
						'bert':0, 'twitter':0, 'elmo':0, 'roberta':1}
eval(model, selected_embeddings, corpus)