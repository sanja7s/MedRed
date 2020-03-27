from flair.data import Corpus
from flair.data import Sentence 

from flair.embeddings import TokenEmbeddings, WordEmbeddings, \
      StackedEmbeddings, CharacterEmbeddings, FlairEmbeddings, \
      PooledFlairEmbeddings, ELMoEmbeddings, BertEmbeddings , RoBERTaEmbeddings
from typing import List

from create_flair_corpus import read_in_AMT, read_in_CADEC, read_in_TwitterADR, read_in_Micromed

# 6. initialize trainer
from flair.trainers import ModelTrainer

# 5. initialize sequence tagger
from flair.models import SequenceTagger

# 6. initialize trainer
from flair.training_utils import EvaluationMetric

# 9. continue trainer at later point
from pathlib import Path

def train(model, selected_embeddings):
  # 1. get the corpus
  if model == 'AMT':
    corpus = read_in_AMT()
  elif model == 'CADEC':
    corpus = read_in_CADEC()
  elif model == 'TwitterADR':
    corpus = read_in_TwitterADR()
  elif model == 'Micromed':
    corpus = read_in_Micromed()
  print(corpus)

  # 2. what tag do we want to predict?
  tag_type = 'ner'

  # 3. make the tag dictionary from the corpus
  tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
  print(tag_dictionary.idx2item)


  embedding_types: List[TokenEmbeddings] = [
  ]

  if selected_embeddings['glove']:
    embedding_types.append(WordEmbeddings('glove'))

  if selected_embeddings['twitter']:
    embedding_types.append(WordEmbeddings('twitter'))

  if selected_embeddings['char']:
    embedding_types.append(CharacterEmbeddings())

  # FlairEmbeddings
  if selected_embeddings['flair']:
    embedding_types.append(FlairEmbeddings('news-forward'))

  # sFlairEmbeddings
  if selected_embeddings['flair']:
    embedding_types.append(FlairEmbeddings('news-backward'))

  # PooledFlairEmbeddings
  if selected_embeddings['pooled-flair']:
    embedding_types.append(PooledFlairEmbeddings('news-forward', pooling='mean'))

  # PooledFlairEmbeddings
  if selected_embeddings['pooled-flair']:
    embedding_types.append(PooledFlairEmbeddings('news-backward', pooling='mean'))

  # init BERT
  if selected_embeddings['bert']:
    embedding_types.append(BertEmbeddings())

  # init roberta
  if selected_embeddings['roberta']:
    embedding_types.append(RoBERTaEmbeddings())

    # init  BioBERT
  if selected_embeddings['biobert']:
    embedding_types.append(BertEmbeddings("data/embeddings/biobert-pubmed-pmc-cased"))

  # init clinical BERT
  if selected_embeddings['clinicalbiobert']:
    embedding_types.append(BertEmbeddings("data/embeddings/pretrained_bert_tf/biobert-base-clinical-cased"))


  # init multilingual ELMo
  if selected_embeddings['elmo']:
    embedding_types.append(ELMoEmbeddings())



  embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)



  tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                          embeddings=embeddings,
                                          tag_dictionary=tag_dictionary,
                                          tag_type=tag_type,
                                          use_crf=True
                                          )



  trainer: ModelTrainer = ModelTrainer(tagger, corpus)

  selected_embeddings_text = [key  for key in selected_embeddings if selected_embeddings[key]]
  selected_embeddings_text = '_'.join(selected_embeddings_text)

  model_dir = 'resources/taggers/FA_' + model + selected_embeddings_text

  # 7. start training
  trainer.train(model_dir,
                train_with_dev=True,
                learning_rate=0.1,
                mini_batch_size=4,
                max_epochs=200,
                checkpoint=True)

  # 8. plot training curves (optional)
  from flair.visual.training_curves import Plotter
  plotter = Plotter()
  plotter.plot_training_curves(model_dir + '/loss.tsv')
  plotter.plot_weights(model_dir + '/weights.txt')



def test(model, selected_embeddings):
  selected_embeddings_text = [key  for key in selected_embeddings if selected_embeddings[key]]
  selected_embeddings_text = '_'.join(selected_embeddings_text)

  print (selected_embeddings_text)

  model_dir = 'resources/taggers/' + model + selected_embeddings_text + '_fine-tuned7s'

  # load the model you trained
  model = SequenceTagger.load(model_dir + '/best-model.pt')

  sentence = Sentence("If you've been on a low calorie diet + exercise for a long time, probably you have low free T3 blood levels causing your hypo symptoms. You should ask specifically for freeT3 and freeT4 to be tested. The low conversion of T4 to T3 is your bodies way of ""protecting itself"" from any further calorie deficiet. The rest of this only matters if you do get low T3 confirmed: it is important you do not go on a T4 monotherapy, it would very likely make your situation worse because it's tricking your brain into thinking you have more then enough thyroid hormones, while your T3 deficit worsens. Either get T3 and T4 combination or no medication. Instead make sure you have enough Iodine, Selenium and Zinc in your diet and consider significantly increasing your calorie intake! It seems paradoxical but because this will eventually increase you T3 levels and basal metabolic rate it will not necessarily make you gain weight in the long term. Also dizzy spells could be low blood sugar (even if you don't who the classical symptoms of shaking/sweating.) If it is low blood sugar you need to be careful with that and make sure to get some glucose quick (both for preventing your dizzines causing accidents and also because every hypoglycemic state will stress out your metabolic system, autoamplifying the low T3)")

  # # predict tags and print
  model.predict(sentence)

  print(sentence.to_dict(tag_type='ner'))
for x in range(1,10):
  pass

def resume(model1, selected_embeddings, model2):

  # 1. get the corpus
  if model2 == 'AMT':
    corpus = read_in_AMT()
  elif model2 == 'CADEC':
    corpus = read_in_CADEC()
  elif model2 == 'TwitterADR':
    corpus = read_in_TwitterADR()
  elif model2 == 'Micromed':
    corpus = read_in_Micromed()
  print(corpus)

  # 2. what tag do we want to predict?
  tag_type = 'ner'

  # 3. make the tag dictionary from the corpus
  tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
  print(tag_dictionary.idx2item)


  embedding_types: List[TokenEmbeddings] = [
  ]

  if selected_embeddings['glove']:
    embedding_types.append(WordEmbeddings('glove'))

  if selected_embeddings['twitter']:
    embedding_types.append(WordEmbeddings('twitter'))

  if selected_embeddings['char']:
    embedding_types.append(CharacterEmbeddings())

  if selected_embeddings['flair']:
    embedding_types.append(FlairEmbeddings('news-forward'))

  if selected_embeddings['flair']:
    embedding_types.append(FlairEmbeddings('news-backward'))


  if selected_embeddings['pooled-flair']:
    embedding_types.append(PooledFlairEmbeddings('news-forward', pooling='mean'))

  if selected_embeddings['pooled-flair']:
    embedding_types.append(PooledFlairEmbeddings('news-backward', pooling='mean'))

  # init multilingual BERT
  if selected_embeddings['bert']:
    embedding_types.append(BertEmbeddings())


  # init multilingual ELMo
  if selected_embeddings['elmo']:
    embedding_types.append(ELMoEmbeddings())



  embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)


  # tagger: SequenceTagger = SequenceTagger(hidden_size=256,
  #                                         embeddings=embeddings,
  #                                         tag_dictionary=tag_dictionary,
  #                                         tag_type=tag_type,
  #                                         use_crf=True)

  selected_embeddings_text = [key  for key in selected_embeddings if selected_embeddings[key]]
  selected_embeddings_text = '_'.join(selected_embeddings_text)
  model_dir1 = 'resources/taggers/to_resume_CoNLL-03_' + model1 + selected_embeddings_text


  #checkpoint = tagger.load_checkpoint(Path(model_dir1+ '/checkpoint.pt'))
  #trainer = ModelTrainer.load_from_checkpoint(checkpoint, corpus)

  best_model = SequenceTagger.load(Path(model_dir1+ '/best-model.pt'))

  trainer: ModelTrainer = ModelTrainer(best_model, corpus)


  # resources/taggers/to_resume_CADECglove_char_flair/

  model_dir2 = 'resources/taggers/train_with_dev_from_' + model1 + '_to_' + model2 + selected_embeddings_text + '_fine-tuned7s'

  trainer.train(model_dir2,
                EvaluationMetric.MICRO_F1_SCORE,
                train_with_dev=True,
                learning_rate=0.1,
                mini_batch_size=8,
                max_epochs=150,
                checkpoint=True)



# params
model = 'Micromed'
selected_embeddings = {'glove':1, 'char':0, 'flair':0, 'pooled-flair':0, \
                        'bert':1, 'twitter':0, 'elmo':0, 'roberta':0, \
                        'biobert':0, 'clinicalbiobert':0}
train(model, selected_embeddings)

# to train different models and parameteres, you can uncomment and/or change the code below
# # params
# model = 'CADEC'
# selected_embeddings = {'glove':0, 'char':0, 'flair':0, 'pooled-flair':1, \
#                         'bert':0, 'twitter':0, 'elmo':1, 'roberta':1}
# train(model, selected_embeddings)

# # params
# model = 'Micromed'
# selected_embeddings = {'glove':0, 'twitter':1,  'char':0, 'flair':0, 'pooled-flair':1}
# train(model, selected_embeddings)


# selected_embeddings = {'glove':1, 'char':1, 'flair':1}
# model1 = 'CADEC'
# model2 = 'AMT'
# resume(model1, selected_embeddings, model2)


# model = 'AMT'
# selected_embeddings = {'glove':1, 'char':0, 'flair':0, 'pooled-flair':1}
# train(model, selected_embeddings)


# selected_embeddings = {'glove':1, 'char':0, 'flair':0, 'pooled-flair':1}
# model1 = 'CADEC'
# model2 = 'AMT'
# resume(model1, selected_embeddings, model2)


# selected_embeddings = {'glove':1, 'char':0, 'flair':0, 'pooled-flair':1}
# model1 = 'CADEC'
# model2 = 'Micromed'
# resume(model1, selected_embeddings, model2)



