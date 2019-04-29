'''
Matthew Perez
Lexical Feature Extraction
Fillers
Pauses
Speech rate
GoP
'''


import sys
import numpy as np
import pandas as pd
import os
import shutil
from sklearn.feature_selection import mutual_info_classif
import operator

labels = {'82697': 0, '78971': 1, '95465': 0, '13691': 0, '82975': 0, '76883': 1, '23789': 0,
'87083': 0, '71834': 0, '78080': 0, '69695': 1, '68117': 0, '69879': 0, '14630': 1, '07729': 1,
'75256': 0, '59826': 1, '11739': 0, '84596': 1, '88947': 0, '47647': 0, '25066': 0, '52053': 0,
'16486': 0, '40216': 0, '18771': 0, '52080': 1, '45758': 0, '01634': 1, '91117': 0, '46352': 0,
'20221': 1, '26753': 0, '29735': 0, '61496': 0, '69573': 1, '29758': 0, '47939': 0, '58812': 0,
'44209': 0, '07920': 0, '68736': 1, '05068': 0}

class spk(object):
	def __init__(self, id, df):
		self.id = id
		self.datafr = df
		self.utts = []
		self.pauses = []
		self.fillers = []
		self.errors = []
		self.phones = []
		self.duration = 0.0
		self.words = []
	def add_utt(self, utt):
		self.utts.append(utt)
		self.pauses.extend(utt.pauses)
		self.fillers.extend(utt.fillers)
		self.phones.extend(utt.phones)
		self.errors.extend(utt.errors)
		self.words.extend(utt.words)
		self.duration += utt.time
	def gen_feats(self):
		spk_stats = {}
		utt_stats = []
		for utt in self.utts: #Utterance level stats
			utt_stats.append(utt.stats)

		#Get dataframe object from self
		df = self.datafr

		#Get speaker level features (Max, min, range, mean, std, quartile, etc.) #8 feats
		# print('df.shape', df)
		#exit()
		for col in df:
			if col == 'label':
				spk_stats[col]=df[col].median() #all labels should be the same
				continue
			min_tag = 'min_'+col #min
			max_tag = 'max_'+col #max
			range_tag = 'range_'+col #range
			std_tag = 'std_'+col
			mean_tag='mean_'+col
			median_tag='median_'+col
			two_five_tag = '25th_per_'+col
			fifty_tag = '50th_per_'+col
			seven_five_tag = '75th_per_'+col

			spk_stats[min_tag]=df[col].min()
			spk_stats[max_tag]=df[col].max()
			spk_stats[range_tag]= df[col].max() - df[col].min()
			spk_stats[std_tag]=df[col].std()
			spk_stats[mean_tag]=df[col].mean()
			spk_stats[median_tag]=df[col].median()
			spk_stats[two_five_tag]=np.percentile(df[col], 25, interpolation='midpoint')
			spk_stats[fifty_tag]=np.percentile(df[col], 50, interpolation='midpoint')
			spk_stats[seven_five_tag]=np.percentile(df[col], 75, interpolation='midpoint')
		#num feats = 9*38 = 342 + 1 (label)
		#Spk stats = 343x61 (feats x speakers)
		return spk_stats, utt_stats


def _word(line):
	if len(line) == 5 and line[4] != 'sil':
		return line[4]
	return None
def _phone(line):
	return line[2]
def _dur(line):
	return (float(line[1]) - float(line[0])) / 100

class utt(object):
	def __init__(self, lst):
		self.id = lst[0][1:-1] #grab utt_id and Remove quotes
		self.spkr = self.id[:5]
		lst = lst[1:]
		self.duration = [] #utterance duration
		self.phones_name = []
		self.pauses = []
		self.errors = []
		self.fillers = []
		self.words = []
		self.phones = []
		self.word_repetitions = []
		self.phone_repetitions = []
		if self.spkr not in labels.keys():
			assert "ID not in Labels. No label assigned"
		self.label = labels[self.spkr]

		for i in xrange(len(lst)):
			line_arr = lst[i].split(' ')
			if len(line_arr) < 4: #Check input to ensure length is 4
				assert False

			#Parse line input
			wrd = _word(line_arr)
			pne = _phone(line_arr)
			dur = _dur(line_arr) #phone duration
			self.duration.append(dur) #utterance duration

			if wrd is not None:
				if not wrd.startswith('<'):
					#Normal word
					self.words.append(wrd)
				elif wrd.startswith('<') and len(wrd) > 20:
					#Mispronunciation (custom tag)
					self.errors.append(dur)
			#Pauses, Fillers, Errors, Phones
			if pne == 'sil':
				self.pauses.append(dur)
			elif pne == 'flr' or pne == 'lau' or pne == 'brth':
				self.fillers.append(dur)
			elif pne == 'spn' or pne == 'unc':
				self.errors.append(dur)
			else:
				self.phones_name.append(pne)
				self.phones.append(dur)

		self.time = sum(self.duration)
		self.stats = self.__gen_stats()

	def __gen_stats(self):
		assert self.duration != 0
		stats = {}
		stats['label'] = self.label
		stats['words_per_sec'] = 1.0 * len(self.words) / self.time
		stats['words_per_utt'] = 1.0 * len(self.words)
		stats['phones_per_utt'] = 1.0 * len(self.phones)
		stats['phones_per_sec'] = 1.0 * len(self.phones) / self.time
		stats['phones_per_word'] = 1.0 * len(self.phones) / len(self.words)
		stats['fillers_per_utt'] = 1.0 * len(self.fillers)
		stats['fillers_per_sec'] = 1.0 * len(self.fillers) / self.time
		stats['fillers_per_word'] = 1.0 * len(self.fillers) / len(self.words)
		stats['fillers_per_phone'] = 1.0 * len(self.fillers) / len(self.phones)
		stats['fillers_dur'] = 1.0 * sum(self.fillers)
		stats['fillers_portion'] = 1.0 * sum(self.fillers) / self.time
		stats['pauses_per_utt'] = 1.0 * len(self.pauses)
		stats['pauses_per_sec'] = 1.0 * len(self.pauses) / self.time
		stats['pauses_per_word'] = 1.0 * len(self.pauses) / len(self.words)
		stats['pauses_per_phone'] = 1.0 * len(self.pauses) / len(self.phones)
		stats['pauses_dur'] = 1.0 * sum(self.pauses)
		stats['pauses_portion'] = 1.0 * sum(self.pauses) / self.time

		#Compute broad statistics
		stats.update(_stats('phones', self.phones))
		stats.update(_stats('fillers', self.fillers))
		stats.update(_stats('pauses', self.pauses))

		return stats

def _stats(name, lst): #Statistics on Timing information regarding pauses/fillers/phones (utterance-level)
	if len(lst) == 0:
		return {name + '_mean': 0.0, name + '_stdev': 0.0, name + '_0th_per': 0.0, name + '_25th_per': 0.0, name + '_50th_per': 0.0, name + '_75th_per': 0.0, name + '_100th_per': 0.0}
	return {name + '_mean': np.mean(lst), name + '_stdev': np.std(lst), name + '_0th_per': np.min(lst),\
		   name + '_25th_per': np.percentile(lst, 25, interpolation='midpoint'),\
		   name + '_50th_per': np.percentile(lst, 50, interpolation='midpoint'),\
		   name + '_75th_per': np.percentile(lst, 75, interpolation='midpoint'), name + '_100th_per': np.max(lst)}



def GOP_kNN(gop_root_feat, gop_test_list):
	#go through root_feat list and grab gop_scores for each subset. put in dict
	gop_AM_list = ['/gop_train_HC/', '/gop_train_Pre/', '/gop_train_Early/', '/gop_train_Late/']
	gop_class = [0, 1, 2, 3]
	gop_AM_dict = {}
	for AM, lbl in zip(gop_AM_list, gop_class):
		gop_AM_path = gop_root_feat+AM+'gop.1'


		#extract gop
		f = open(gop_AM_path)
		lines = f.readlines()
		lines = [l.split() for l in lines]
		if len(lines) == 0:
			print i
			assert False
		gop_test_data_list=[]
		for i in lines:
			utt_loc = i[0]
			spkr = utt_loc[:5]
			utt_id = utt_loc[:9]
			# new_utt = utt()
			data = [float(j) for j in i[2:-2]] #Get GOP data
			if lbl not in gop_AM_dict.keys():
				gop_AM_dict[lbl] = []
			gop_AM_dict[lbl].append(data) #Puts data in order of gop_AM_list
		gop_AM_dict[lbl]= np.array(gop_AM_dict[lbl])


	#Run kNN on dict
	#def knn(k, X_tr, y_tr, X_te):
	dists = {}
	gop_vector_result = [] #indexed by line (in order). GoP vector for utt
	gop_label_result = [] #indexed by line (in order). GoP bin for utt
	# for AM_label in gop_AM_dict.keys():
	for line in range(len(gop_test_list)): #go through each utterance-line
		for AM_label in gop_AM_dict.keys(): #go through all the bin'd gop models
			print(line, AM_label)
			#print(gop_AM_dict)
			# print('test', len(gop_test_list[line]))
			# print('train', len(gop_AM_dict[AM_label][line]))

			#find closest distance to gop_test (utterance-line in question)
			dists[AM_label] = np.linalg.norm(gop_AM_dict[AM_label][line]) #compute linalg norm to find smallest distance (most similar pronunciation)

		dists = sorted(dists.items(), key=operator.itemgetter(1)) #Sort by distance
		nbrs = [key for (key, v) in dists] #Get array location
		
		closest_neighbor = nbrs[0]
		closest_neighbor_vec = dists[closest_neighbor]

		print(dists)
		exit()


		gop_label_result.append(closest_neighbor)
		gop_vector_result.append(closest_neighbor_vec)
		

	#return prediction
	#knbrs_preds = y_tr[knbrs]


def remove_0_var_IG_utt(df, df_labels):
	drop_list = []
	#print('df pre', df.shape)
	# df_labels = df['label'] #needed for mutual inforamtion calculation
	# df_no_label = df.drop(['label'], 1) #drop labels col


	#remove 0 var
	for col in df:
		if np.var(df[col])==0:
			drop_list.append(col)
	#Remove 0 IG
	mi = mutual_info_classif(df, df_labels)
	for idx, val in enumerate(mi):
		if val<=0 and df.columns[idx] not in drop_list:
			drop_title = df.columns[idx]
			drop_list.append(drop_title)
	
	#return culled dataframe
	return df.drop(drop_list, 1)



def compute_GOP_stats(data):
	gop_dict={}
	gop_dict['mean_gop'] = np.mean(data)
	gop_dict['std_gop'] = np.std(data)
	gop_dict['median_gop'] = np.median(data)
	gop_dict['max_gop'] = np.max(data)
	gop_dict['min_gop'] = np.min(data)
	gop_dict['75_gop'] = np.percentile(data, 75, interpolation='midpoint')
	gop_dict['50_gop'] = np.percentile(data, 50, interpolation='midpoint')
	gop_dict['25_gop'] = np.percentile(data, 25, interpolation='midpoint')
	#Update stats dictionary

	return gop_dict

# def sane(lst):
# 	for i in lst:
# 		print('i', i)
# 		if len(i.split(' ')) == 5 and not i.split(' ')[4].startswith('<') and i.split(' ')[2] != 'sil':
# 			return True
# 	return False



def Normalize(df):
	#Compute mean and std and return arrays
	mean_arr = []
	std_arr = []
	for col in df:
		mean_arr.append(df[col].mean())
		std_arr.append(df[col].std())

	return mean_arr, std_arr

def read_data(word_ali_doc):
	utt_curr = []
	utts_stats=[]
	label_arr=[]
	ids_arr=[]
	with open(word_ali_doc,'r') as word_ali:
		for line in word_ali:
			line=line.strip()

			#check for new utterance
			if line != ".":
				utt_curr.append(line)
			else:
				#write old utt_curr to 
				new_utt = utt(utt_curr)
				utts_stats.append(new_utt.stats)
				#spkr_id = new_utt.id[:5]
				ids_arr.append(new_utt.id)

				utt_curr = []

	#Create Dataframe
	df_data = pd.DataFrame(utts_stats, index=ids_arr)
	return df_data, label_arr

def read_gop(utts_stats, gop_path):
	gop_stats=[]
	gop_stats_ids=[]
	gop_columns = ['mean_gop', '50_gop', '75_gop', 'min_gop', 'median_gop', '25_gop', 'std_gop', 'max_gop']
	
	with open(gop_path, 'r') as r:
		for line in r:
			line_arr = line.split()
			utt_id = line_arr[0]
			spkr_id = utt_id[:5]
			gop_data = np.array(line_arr[2:-1]).astype(np.float) #ignore utt_id, and surrounding brackets
			#Put dict of gop values in list
			gop_stats.append(compute_GOP_stats(gop_data))
			gop_stats_ids.append(utt_id)

	df_gop = pd.DataFrame(gop_stats, index=gop_stats_ids)
	df_feat = pd.concat([utts_stats, df_gop], axis=1)
	#return all feats
	return df_feat


def extract_features(word_ali_doc, gop_data_path):
	#Var init.
	utts_stats=[]
	label_arr=[]

	df_data, label_arr = read_data(word_ali_doc) #get lex features
	df_data = read_gop(df_data, gop_data_path) #combine with gop features

	#Remove 0 var and IG
	df_labels = df_data['label']
	df_no_label = df_data.drop(['label'], 1)
	df_no_label = remove_0_var_IG_utt(df_no_label, df_labels)

	#Get mean and std and normalize from training mean and std
	mean_arr, std_arr = Normalize(df_no_label)
	for idx, col in enumerate(df_no_label):
		df_no_label[col] = (df_no_label[col] - mean_arr[idx]) / std_arr[idx]
	#Add labels back in
	df_data = pd.concat([df_labels, df_no_label], axis=1)
	# print(df_data.shape)
	return df_data


def main():
	if len(sys.argv) != 5:
		print('Need ali_path, val/train, gop path, feat_doct_path')
		exit()
	ali_path = sys.argv[1]
	te_tr = sys.argv[2]
	gop_data_path = sys.argv[3]
	feat_doc_path = sys.argv[4]

	word_ali_doc = ali_path+'/word_ali_'+te_tr+'.txt'
	#feat_doc = data_path+'/lexical_features/feat_'+te_tr+'.csv'
	feat_doc = feat_doc_path+'/feat_'+te_tr+'.csv'
	df_features = extract_features(word_ali_doc, gop_data_path)

	#write features
	df_features.to_csv(feat_doc)


if __name__ == '__main__':
	main()
