import pandas as pd
import numpy as np
from collections import defaultdict, Counter, OrderedDict
import re
import nltk
from nltk import word_tokenize
# THIS needs to be RUN once, for the first time
# nltk.download('punkt')
import string
"""
	a bit low-level code to transform the AMT inputs to NER labels for DL
"""

class CADEC_labels():


	def __init__(self):

		self.fin = '../../data/AMT/cadec_labels.csv'
		self.fout = '../../data/AMT/labels/NER_cadec_labels.csv'
		
		self.fout2 = '../../data/AMT/labels/filtered_cadec_labels.csv'

	def __parse_answers(self, df, column_name):

		# maybe just do NLTK tokenize here
		def parse_row(row):
			row = [ elem.strip().strip(string.punctuation) for elem in row ]
			return [ elem for elem in row if elem != '' ]
		df[column_name] = df[column_name].apply(
		    #lambda row: row.split(';')
		    lambda row: re.split(';|,|/|\.',str(row) )
			)
		df[column_name] = df[column_name].apply(
		    lambda row: parse_row(row)
			)


	def assign_tags_to_text(self, text, entity_type='DIS', entities=None):

		# print(text)

		# sort the entities by size (length of tokens)
		# because we will prioritise longer among overlapping entities
		entities = sorted(entities, key=lambda x: len(x.split()), reverse=True)
		
		# this is to check how many entities we end up discarding in this way
		cnt_not_found = 0
		# text into tokens -- we removed punctuation in AMT, so here, too
		try:
			split_text = word_tokenize(text)
		except TypeError:
			print (text, entities)
			return {}, [], 0, 0

		# print(split_text)

		# let us create two indices for each token -- one is its position in the list
		# the other is the character position in the text
		# the first index is used to mark with the tags the entity tokens
		# the second index will be used to compare the output of str.search
		indices_TAGS = OrderedDict({ (i, token):'O' for i, token in enumerate( split_text ) })
		assert(len(indices_TAGS) == len(split_text))


		double_indices_TAGS_lst = OrderedDict()
		double_indices_TAGS = OrderedDict()
		# go through the split text tokens and search for all the indices where they are found
		# we need to assign to each token their right position using the double indices
		for (i,token) in indices_TAGS:
			try:
				all_ii_found = [ m.start() for m in re.finditer(re.escape(token), text) ]
				#print(all_ii_found)
				all_ii = frozenset(all_ii_found)
			except:
				all_ii = frozenset([])

			# we save for each token, its position among the tokens, 
			# and the index of its firsct character in the original text 
			double_indices_TAGS_lst[(i, all_ii, token)] = 'O'

		assert(len(double_indices_TAGS_lst) == len(split_text))

		# we now select from the list of possible indices for each token
		# the right one -- because we know its token's position, 
		# i.e., the first index in the pair
		prev_ii = -1
		for (i, all_ii, token) in double_indices_TAGS_lst:
			all_ii_lst = sorted(list(all_ii))
			#print(tag, all_ii_lst)
			if all_ii_lst == []:
				this_ii = prev_ii+1
			for possible_ii in all_ii_lst:
				if possible_ii > prev_ii:
					this_ii = possible_ii
					break
			prev_ii = this_ii
			double_indices_TAGS[(i, this_ii, token)] = 'O'

		assert(len(double_indices_TAGS) == len(split_text))

			
		entity_missmatch_with_AMT = 0
		entity_missmatched = []
		kept_entities = []

		# go from the longest to the shortest entities
		for e in entities:
			if e == 'nan':
				continue
			found_i = -1
			# if they are found in the raw text
			if text.find(e) != -1:
				# get the index of its FIRST position
				#found_ii = text.find(e)
				found_ii_list = [ m.start() for m in re.finditer(re.escape(e), text) ]
				l = len(e.split())
				# now find the corresponding tokens
				for (i, ii, token) in double_indices_TAGS:
					if ii in found_ii_list:
						# mark the first token, i.e., the beginning
						found_i = i
						if indices_TAGS[(found_i, token)] == 'O': 
							indices_TAGS[(found_i, token)] = 'B-' + entity_type
							kept_entities.append(e)
							# find the rest of the tokens and mark them with inside tags	
							s = 1
							while s < l:
								if indices_TAGS[( found_i+s, split_text[found_i+s] )] == 'O':
									indices_TAGS[( found_i+s, split_text[found_i+s] )] = 'I-' + entity_type
								else:
									print ('XXXXX  UNSOLVED OVERLAPP  XXXXX')
								s += 1
							break
				# this is to be solved -- how you clean AMT text and parse tokens here should be the same
				if found_i == -1:
					entity_missmatch_with_AMT += 1
					entity_missmatched.append(e)
					continue

				
				#print (e, found_ii, found_i+s+1)
			else:
				#print ('NOT FOUND', e)
				cnt_not_found += 1
				
		# if entity_missmatch_with_AMT or cnt_not_found:
		# 	print(entity_missmatch_with_AMT, cnt_not_found)
		# 	print(entity_missmatched)
		# 	unkept_entities = set(entities).difference(set(kept_entities))
		# 	print(unkept_entities)
		return indices_TAGS, kept_entities, entity_missmatch_with_AMT, cnt_not_found


	def parse_all(self):

		self.df_in = pd.read_csv(self.fin)
		self.__parse_answers(self.df_in, 'golden.symptoms0')
		self.__parse_answers(self.df_in, 'golden.drugs0')

		df_out = self.df_in.copy()

		with open(self.fout, 'w') as f:

			total_entities, total_missmatched_entities, total_unkept_entities = 0, 0.0, 0.0

			all_sym_kept, all_drug_kept = [], []

			for i, line in self.df_in.iterrows():
				syms = line['golden.symptoms0']
				drugs = line['golden.drugs0']
				text = line['body']


				# if i in range(439,459):
				# 	print (i, text),
				# 	print (syms)
				# 	print (drugs)

				#print(i)
				post_tags_sym, sym_kept, sym_missmatch_with_AMT, sym_unkept = self.assign_tags_to_text(text, entity_type='DIS', entities=syms)
				#print(sym_kept)
				post_tags_drug, drug_kept, drug_missmatch_with_AMT, drug_unkept = self.assign_tags_to_text(text, entity_type='DRUG', entities=drugs)
				# print(drug_kept)
				post_tags = OrderedDict({ (index, token): sym_tag if sym_tag != 'O' else post_tags_drug[(index, token)] for (index, token), sym_tag in post_tags_sym.items() })

				all_sym_kept.append(';'.join(sym_kept))
				all_drug_kept.append(';'.join(drug_kept))

				total_entities += len(sym_kept) + len(drug_kept)
				total_missmatched_entities += sym_missmatch_with_AMT + drug_missmatch_with_AMT
				total_unkept_entities += sym_unkept + drug_unkept

				i_post_len = 0
				for (index, token),tag in post_tags.items():
					if token != 'null':
						i_post_len += 1
						f.write(token + '\t' + tag + '\n')

						if i_post_len >= 300:
							i_post_len = 0
							print ("For long posts ")
							f.write('\t\n')

				# this is to match CADEC format, one post and then an empty line
				f.write('\t\n')

				# if i == 1:
				# 		break
				#print (post_tags)
			print("Processed {} posts. Percent of entities not matched {:.2f}%, and percent of entities discounted {:.2f}%, of total {} accepted.".\
				format(i, total_missmatched_entities/total_entities*100, total_unkept_entities/total_entities*100, total_entities))

		df_out['new_sym'] = all_sym_kept
		df_out['new_drug'] = all_drug_kept

		df_out.to_csv(self.fout2, columns=['golden.symptoms0', 'new_sym','golden.drugs0', 'new_drug', 'body'])


	def split_train_test(self):

		# might be able to skip this -- just do the merging later as done for words, anyway
		#self.remove_wrong_chars()

		all_labels = pd.read_csv(self.fout, sep='\t', header=None)
		N = len(all_labels)
		train = all_labels.iloc[:int(N*0.6)]
		#train.loc[len(train)] = ["-DOCEND-", 'O']
		dev = all_labels.iloc[int(N*0.6):int(N*0.80)]
		#dev.loc[len(dev)] = ["-DOCEND-", 'O']
		test = all_labels.iloc[int(N*0.80):]
		#test.loc[len(test)] = ["-DOCEND-", 'O']
		print (N, len(train), len(dev), len(test))
		assert (len(train)+len(test)+len(dev) == N)
		train.to_csv(self.fout.replace('.csv', '_train.csv'), sep=' ', index=None, header=None)
		dev.to_csv(self.fout.replace('.csv', '_dev.csv'), sep=' ', index=None, header=None)
		test.to_csv(self.fout.replace('.csv', '_test.csv'), sep=' ', index=None, header=None)

	# not needed and not used after all
	def remove_wrong_chars(self):

		cadec_chars, cadec_tags = [], []

		cadec_chars_file = '../data/NER/preprocessed/chars.txt'
		with open(cadec_chars_file, 'r') as f:
			for line in f:
				cadec_chars.append(line.strip())

		cadec_tags_file = '../data/NER/preprocessed/tags.txt'
		with open(cadec_tags_file, 'r') as f:
			for line in f:
				cadec_tags.append(line.strip())

		print ("Total cadec chars ", len(cadec_chars))
		cadec_set = set(cadec_chars)

		all_labels = pd.read_csv(self.fout, sep='\t', header=None)
		clean_labels = all_labels.copy()
		print (all_labels.head())

		to_delete = []

		for i, row in all_labels.iterrows():
			word_set = set(list(str(row.values[0])))
			# if word_set.difference(cadec_set):
			# 	print (row.values, word_set.difference(cadec_set))
			# 	to_delete.append(i)
			# if len(word_set) == 0:
			# 	print ("~~~~~", i)
			# 	to_delete.append(i)

			# if not (str(row.values[1]) in cadec_tags):
			# 	print ("^^^^^^^^^^^^", i)
			# 	to_delete.append(i)



		print (to_delete)

		clean_labels = clean_labels.drop(clean_labels.index[to_delete])

		print ("Dropped {} words out of total {}.".format( len(all_labels)-len(clean_labels), len(all_labels) ))

		clean_labels = clean_labels.reset_index(drop=True)

		clean_labels.to_csv(self.fout, sep='\t', header=None, index=None)

amt = CADEC_labels()
amt.parse_all()
# amt.remove_wrong_chars()
amt.split_train_test()