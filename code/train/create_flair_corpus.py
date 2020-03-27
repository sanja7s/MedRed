from flair.data import Corpus
from flair.datasets import ColumnCorpus


def read_in_CADEC_prev():
	# define columns
	columns = {0: 'text', 1: 'text_lower', 2: 'pos', 3: 'ner', 4: 'text_start', 5: 'text_end', 6: 'SNOMEDCT_id'}

	# this is the folder in which train, test and dev files reside
	data_folder = 'data/CADEC/NER/'

	# init a corpus using column format, data folder and the names of the train, dev and test files
	corpus: Corpus = ColumnCorpus(data_folder, columns,
	                              train_file='corpus-conll-train.txt',
	                              test_file='corpus-conll-test.txt',
	                              dev_file='corpus-conll-dev.txt')

	print(corpus)

	print(corpus.train[0].to_tagged_string('pos'))
	print(corpus.train[0].to_tagged_string('ner'))

	return corpus

def read_in_CADEC():
	# define columns
	columns = {0: 'text', 1: 'ner'}

	# this is the folder in which train, test and dev files reside
	data_folder = 'data/CADEC/NER7s/'

	# init a corpus using column format, data folder and the names of the train, dev and test files
	corpus: Corpus = ColumnCorpus(data_folder, columns,
	                              train_file='NER_cadec_labels_train.csv',
	                              test_file='NER_cadec_labels_test.csv',
	                              dev_file='NER_cadec_labels_dev.csv')

	len(corpus.train)
	print(corpus)

	# print(corpus.train[0].to_tagged_string('pos'))
	print(corpus.train[0].to_tagged_string('ner'))

	print(corpus.make_tag_dictionary('ner'))

	return corpus


def read_in_AMT():
	# define columns
	columns = {0: 'text', 1: 'ner'}

	# this is the folder in which train, test and dev files reside
	data_folder = 'data/AMT/'

	# init a corpus using column format, data folder and the names of the train, dev and test files
	corpus: Corpus = ColumnCorpus(data_folder, columns,
	                              train_file='NER_Reddit_AMT_labels_train.csv',
	                              test_file='NER_Reddit_AMT_labels_test.csv',
	                              dev_file='NER_Reddit_AMT_labels_dev.csv')

	len(corpus.train)

	# print(corpus.train[0].to_tagged_string('pos'))
	print(corpus.train[0].to_tagged_string('ner'))

	print (corpus)

	print(corpus.make_tag_dictionary('ner'))

	return corpus


def read_in_TwitterADR():
	# define columns
	# avelox-51c3e5a853785f584a9a8c01	76	93	ADR	connective tissue	avelox	avelox
	columns = {0: 'text', 1: 'ner'}

	# this is the folder in which train, test and dev files reside
	data_folder = 'data/TwitterADR/'

	# init a corpus using column format, data folder and the names of the train, dev and test files
	corpus: Corpus = ColumnCorpus(data_folder, columns,
	                              train_file='TwitterADR_train.csv',
	                              test_file='TwitterADR_test.csv',
	                              dev_file='TwitterADR_dev.csv')

	len(corpus.train)

	print(corpus.train[0].to_tagged_string('pos'))
	print(corpus.train[0].to_tagged_string('ner'))

	return corpus
	



def read_in_Micromed():
	# define columns
	# avelox-51c3e5a853785f584a9a8c01	76	93	ADR	connective tissue	avelox	avelox
	columns = {0: 'text', 1: 'ner'}

	# this is the folder in which train, test and dev files reside
	data_folder = 'data/Micromed/'

	# init a corpus using column format, data folder and the names of the train, dev and test files
	corpus: Corpus = ColumnCorpus(data_folder, columns,
	                              train_file='Micromed_train.csv',
	                              test_file='Micromed_test.csv',
	                              dev_file='Micromed_dev.csv')

	len(corpus.train)

	print(corpus.train[0].to_tagged_string('pos'))
	print(corpus.train[0].to_tagged_string('ner'))

	return corpus


# read_in_CADEC()
# read_in_Micromed()
# read_in_AMT()