import os
import numpy as np
import sys
from sys import stdout
from sys import path
import re
import hashlib
import chaipy.common as commons
import chaipy.praat as praat
import wave
import shutil
path.append('/z/mkperez/PremanifestHD-classification/Scripts/kaldi/helper') #needed to import clean_transcripts
import clean_transcripts

#0=healthy, 1=pre, 2=early, 3=late
all_labels = {'82697': 0, '78971': 1, '95465': 0, '13691': 0, '82975': 0, '44574': 3, '35063': 3,
'23789': 0, '87083': 0, '71834': 0, '76373': 3, '69879': 0, '78080': 0, '69695': 1, '38717': 2,
'68117': 0, '00359': 2, '81392': 3, '14630': 1, '95407': 3, '07729': 1, '56896': 3, '75256': 0,
'76883': 1, '24371': 2, '78597': 2, '59826': 1, '11739': 0, '18261': 2, '53870': 2, '84596': 1,
'61496': 0, '88947': 0, '47647': 0, '32762': 2, '25066': 0, '52053': 0, '16486': 0, '40216': 0,
'18771': 0, '52080': 1, '45758': 0, '50377': 2, '91117': 0, '07920': 0, '46352': 0, '20221': 1,
'26753': 0, '69573': 1, '73828': 2, '29735': 0, '29758': 0, '47939': 0, '80292': 3, '55029': 2,
'58812': 0, '44209': 0, '42080': 2, '68736': 1, '05068': 0, '33752': 2, '01634': 1}
#31 healthy, 12 premanifest
pre_healthy = {'82697': 0, '78971': 1, '95465': 0, '13691': 0, '82975': 0, '76883': 1, '23789': 0,
'87083': 0, '71834': 0, '78080': 0, '69695': 1, '68117': 0, '69879': 0, '14630': 1, '07729': 1,
'75256': 0, '59826': 1, '11739': 0, '84596': 1, '88947': 0, '47647': 0, '25066': 0, '52053': 0,
'16486': 0, '40216': 0, '18771': 0, '52080': 1, '45758': 0, '01634': 1, '91117': 0, '46352': 0,
'20221': 1, '26753': 0, '29735': 0, '61496': 0, '69573': 1, '29758': 0, '47939': 0, '58812': 0,
'44209': 0, '07920': 0, '68736': 1, '05068': 0}

def get_textGrid(filelist, spkr):
	for filename in filelist:
		if filename.startswith(spkr) and filename.endswith('.textGrid'):
			return filename
	return "0"

def createWavScp(rawDataPath, kaldiPath, trainSet, valSet):
	trainWavScp = open(kaldiPath + '/train/wav.scp', 'w')
	valWavScp = open(kaldiPath + '/val/wav.scp', 'w')
	filelist = sorted(os.listdir(rawDataPath))
	for filename in filelist:
		if filename.endswith('.wav'):
			rcd = filename.split(".")[0]
			wav_full_path = rawDataPath + '/'+ filename

			if rcd in trainSet:
				trainWavScp.write('%s %s\n' %(rcd, wav_full_path))
			elif rcd in valSet:
				valWavScp.write('%s %s\n' %(rcd, wav_full_path))
	trainWavScp.close()
	valWavScp.close()

def createData(raw_data_path, writePath, lexiconPath, item_set):
	segments = open(writePath + '/segments', 'w')
	textFile = open(writePath + '/text', 'w')
	utt2spk = open(writePath + '/utt2spk', 'w')
	oov_dict = {}
	lexicon = [x.strip().upper().split('\t')[0] for x in open(lexiconPath).readlines()]
	
	for spkr in item_set:
		filelist = sorted(os.listdir(raw_data_path))
		filename = get_textGrid(filelist, spkr)
		if filename=="0":
			print("no file found for spkr ", spkr)
			exit()
		intervals = praat.TextGrid.from_file(raw_data_path +'/'+filename).items.items()[0][1].intervals

		print('interval', intervals)
		count = 0
		for interval in intervals:
			text = interval.text
			utt = '%s_%03d' %(spkr, count)
			count += 1

			#extract transcriptions
			ftext, wlabels = clean_transcripts.clean(text,utt,lexicon, oov_dict)

			if isinstance(ftext, tuple) or len(ftext) == 0:
				continue
			# print('ftext', ftext)
			# print('ftext', len(ftext))
			#print('wlabels')
			# for i in xrange(len(wlabels)):
			#     wordLabelText = ftext[int(wlabels[i][0])] + '\t'  +\
			#                                ' '.join(l for l in wlabels[i])
			writeText = ' '.join(w for w in ftext)
			textFile.write('%s %s\n' %(utt, writeText))
			utt2spk.write('%s spk%s\n' %(utt, spkr))
			segments.write('%s %s %f %f\n' %(utt,spkr,interval.get_start_time(),interval.get_end_time()))
			# print('%s %s %f %f\n' %(utt,spkr,interval.get_start_time(),interval.get_end_time()))
			# exit()
	#close files
	segments.close()
	textFile.close()
	utt2spk.close()

	oovfile = open('/z/mkperez/PremanifestHD-classification/Data/global-dict/GF-oov.txt', 'w')
	if oov_dict.has_key('UNIBET'):
		for i in oov_dict['UNIBET'].keys():
			oovfile.write('%s %s\n' %(i, oov_dict['UNIBET'][i][0]))

def create_folds(k_path, num_folds, overwrite):
	#overwrite kaldi dir
	if overwrite=="true":
		if os.path.exists(k_path):
			shutil.rmtree(k_path)
		os.mkdir(k_path)

	#create fold dir
	for i in range(num_folds):
		writePath = k_path+"/set"+str(i)
		#create directories if not created
		if not os.path.isdir(writePath):
			os.mkdir(writePath)
			os.mkdir(writePath + '/train')
			os.mkdir(writePath + '/val')


def main():
	# n_l={}
	# for l in labels.keys():
	#     if labels[l]<=1:
	#         n_l[l] = labels[l]
	# print(n_l, len(n_l.keys()))
	# exit()


	###VARIABLE INIT###
	if len(sys.argv) != 4:
		print("Not enough arguments")
		exit()
	#kaldi_path="/z/mkperez/PremanifestHD-classification/Data/GF-Passage"
	raw_data_path=sys.argv[1]
	kaldi_path=sys.argv[2]
	overwrite=sys.argv[3]
	lexiconPath='/z/mkperez/PremanifestHD-classification/Data/global-dict/lexicon.txt' #make this

	labels = pre_healthy
	###create folds using LOO###
	create_folds(kaldi_path, len(labels.keys()), overwrite)

	###create kaldi data###
	for key, set_num in zip(labels.keys(), range(len(labels.keys()))): #key=participant elem, set_num=fold_num
		val_set=[key]
		train_set=labels.keys()
		train_set.remove(key)
		set_path=kaldi_path+"/set"+str(set_num)

		createWavScp(rawDataPath=raw_data_path, kaldiPath=set_path, trainSet=train_set, valSet=val_set)
		createData(raw_data_path, set_path + '/train', lexiconPath, train_set)
		createData(raw_data_path, set_path + '/val', lexiconPath, val_set)


if __name__ == '__main__':
	main()
