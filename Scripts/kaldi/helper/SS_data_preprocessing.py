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
import io
import subprocess
import sox

healthy_all_late = {'46352': 0, '95465': 0, '13691': 0, '82975': 0, '35063': 3, '29758': 0, 
'87083': 0, '71834': 0, '76373': 3, '78080': 0, '68117': 0, '81392': 3, '95407': 3, '56896': 3, '75256': 0, 
'88947': 0, '47647': 0, '25066': 0, '52053': 0, '16486': 0, '18771': 0, '45758': 0, '91117': 0, '26753': 0, 
'29735': 0, '61496': 0, '23789': 0, '47939': 0, '05068': 0, '80292': 3, '58812': 0, '44209': 0, '07920': 0, 
'44574': 3, '11739': 0}

# Healthy + Premanifest
healthy_premanifest = {'46352': 0, '20221': 1, '78971': 1, '52053': 0, '82975': 0, '69573': 1, 
'23789': 0, '75256': 0, '76883': 1, '29758': 0, '52080': 1, '78080': 0, '69695': 1, '01634': 1, 
'91117': 0, '59826': 1, '87083': 0, '14630': 1, '84596': 1, '71834': 0}

early_late_balanced = {'44574': 3, '95465': 0, '13691': 0, '35063': 3, '11739': 0, '76373': 3,
'38717': 2, '68117': 0, '00359': 2, '81392': 3, '07920': 0, '95407': 3, '56896': 3, '24371': 2, '78597': 2,
'18261': 2, '53870': 2, '61496': 0, '88947': 0, '47647': 0, '32762': 2, '25066': 0, '16486': 0, '18771': 0,
'45758': 0, '50377': 2, '26753': 0, '29735': 0, '73828': 2, '47939': 0, '80292': 3, '55029': 2, '58812': 0,
'44209': 0, '42080': 2, '05068': 0, '33752': 2}

late_balanced = {'47647': 0, '35063': 3, '56896': 3,'26753': 0, '44574': 3, '81392': 3, '18771': 0,
'76373': 3, '45758': 0, '47939': 0, '80292': 3, '58812': 0, '68117': 0, '95407': 3}

def unicodetoascii(text):
	TEXT=text.replace('\xe2\x80\x99', "'")
	return TEXT

def convert_min_to_sec(t):
	# print('t', t)
	#print(t.split(':'))
	minutes = float(t.split(':')[0])
	seconds = float(t.split(':')[1])
	minutes = 60 * minutes
	# print('min', minutes)
	# print('seconds', seconds)
	return minutes+seconds

def get_textGrid(filelist, spkr):
	for filename in filelist:
		if filename.startswith(spkr) and filename.endswith('.txt'):
			return filename
	return "0"

def createWavScp(rawDataPath, kaldiPath, itemSet):
	with open(kaldiPath + '/wav.scp', 'w') as wavSCP:
		for item in itemSet:

			filelist = [s for s in sorted(os.listdir(rawDataPath)) if s.endswith(".wav")]
			indices = [i for i, s in enumerate(filelist) if item in s]
			if indices:
				#print('index ', indices, item)
				wav_full_path = rawDataPath + '/'+filelist[indices[0]] #Get first index (earliest instance of match) and grab wav-file name
				# print('wav_full_path', wav_full_path)
				# exit()

				wavSCP.write('%s %s\n' %(item, wav_full_path))

				# #downsample to 8k
				# wavSCP.write('%s sox %s -t wav -r %d -c 1 - remix 2 |\n' %(item, wav_full_path, 8000))
			else:
				print('nothing found', item)
				exit()

def createData(raw_data_path, writePath, lexiconPath, item_set):
	segments = open(writePath + '/segments', 'w')
	textFile = open(writePath + '/text', 'w')
	everyonetextFile = open(writePath + '/all_text_clinician', 'w') #clinician included
	utt2spk = open(writePath + '/utt2spk', 'w')
	oov_dict = {}
	lexicon = [x.strip().upper().split('\t')[0] for x in open(lexiconPath).readlines()]
	
	#For each spkr
	for spkr in item_set:
		filelist = sorted(os.listdir(raw_data_path))
		filename = get_textGrid(filelist, spkr)
		if filename=="0":
			print("no file found for spkr ", spkr)
			exit()

		count = 0
		#For each line in session
		with open(raw_data_path +'/'+filename, 'r') as r:
			for line in r:
				#Ensure line is not whitespace
				if line.strip():
					line = unicodetoascii(line)
					line_arr = line.rstrip().split()
					print('line', spkr, line_arr)
					# exit()


					#skip participants relative
					if len(line.split(':')[0].split())==2:
						continue

					#Skip clinician voice
					if line_arr[0].startswith("R") or line_arr[0].startswith("["):
						continue

					#Check if time boundaries are there, 
					if line_arr[-2].startswith("[") and line_arr[-2].endswith("]") and \
					line_arr[-1].startswith("[") and line_arr[-1].endswith("]"):

						spkr_id = line_arr[0].replace(":", "")



						#Combine Par and spkr_PIN
						spkr_id += spkr
						utt_id = '%s-%03d' %(spkr_id, count)
						count += 1
						#print('spkr_id', spkr_id)
						# print('end t', end_t)
						# print('start_t', start_t)

						#Omit time boundaries and spkr id
						text = " ".join(line_arr[1:-2])

						#extract transcriptions using clean trasncript
						# ftext, wlabels = clean_transcripts.clean(text,utt_id,lexicon, oov_dict)
						# if isinstance(ftext, tuple) or len(ftext) == 0:
						# 	continue
						# writeText = ' '.join(w for w in ftext) #create write text


						if spkr_id.startswith("p") or spkr_id.startswith("P"): #Participant speech, just write to text file
							#Get timing
							start_t = convert_min_to_sec(line_arr[-2].replace("[", "").replace("]", ""))
							end_t = convert_min_to_sec(line_arr[-1].replace("[", "").replace("]", ""))
							if start_t > end_t:
								print("times misaligned", spkr," ", line)
								exit()
							#Write files
							# textFile.write('%s %s\n' %(utt_id, writeText))
							# utt2spk.write('%s %s\n' %(utt_id, spkr_id))
							segments.write('%s %s %f %f\n' %(utt_id, spkr, start_t, end_t))

						#write to everyoneTextFile
						# everyonetextFile.write('%s %s\n' %(utt_id, writeText))

					else:
						#Skip portions which are not time boundaried (ie not clear enough, overlapping speech, etc.)
						print("SKIP: no end times for spkr", spkr, " on line ", line)
						# exit()


	#close files
	segments.close()
	textFile.close()
	utt2spk.close()

	oovfile = open('/z/mkperez/SpontSpeech-HD/Data/global-dict/SS-oov.txt', 'w')
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

	#all folds
	writePath = k_path+"/all_data_test"
	if not os.path.isdir(writePath):
		os.mkdir(writePath)
		os.mkdir(writePath + '/test')


def putClinicianLMText(raw_data_path, set_path):
	#Augment training all_text_clinician with clinician text
	#To test whether clinician context can improve LM and ultimately ASR
	with open(set_path+"/train/all_text_clinician", 'a') as w:
		with open(set_path+"/val/all_text_clinician", 'r') as r:
			for lines in r:
				line_arr = lines.split()
				utt_id = line_arr[0]
				if utt_id.startswith('r') or utt_id.startswith('R'): #Put clinican text in train text
					print('writing line: {} in training'.format(lines))
					w.write(lines)
					

def segment_wav_files(segment_file, wav_scp_file, new_wav_root):
	# Create segmented wav files using sox
	# Put files in a separate directory
	with open(segment_file, 'r') as seg_file:
		with open(wav_scp_file, 'r') as wav_scp:
			wav_dic = {}
			seg_dic = {} #list of tuples with starting/end time


			for line in wav_scp:
				line_arr = line.split(' ',1)
				# print(line_arr)
				# exit()
				spkr = line_arr[0]
				sox_first_line = line_arr[1]
				wav_dic[spkr] = str(sox_first_line)

			for line in seg_file:
				line_arr = line.split()
				utt_id=""
				if 'PARTICIPANT' in line_arr[0] or 'PARTCIPANT' in line_arr[0]:
					utt_id = line_arr[0].split('T')[-1]
				else:
					utt_id = line_arr[0].split('P')[-1]
				spkr = line_arr[1]
				# print(spkr)
				# exit()
				begin_T = line_arr[2]
				end_T = line_arr[3]
				if spkr not in seg_dic:
					seg_dic[spkr] = []
				seg_dic[spkr].append((utt_id, begin_T, end_T))

	for spkr in seg_dic:
		#first_line = wav_dic[spkr].strip()[:-1]
		in_wav = wav_dic[spkr].strip()
		# print(first_line)
		# exit()
		for seg in seg_dic[spkr]:
			# print(seg)
			utt_id = seg[0]
			beg_t = seg[1]
			end_t = seg[2]

			out_wav = os.path.join(new_wav_root,utt_id+'.wav')

			transformer = sox.Transformer()
			transformer.trim(float(beg_t),float(end_t))
			transformer.convert(samplerate=32000, n_channels=1)
			transformer.build(in_wav, out_wav)

def main():
	#sanity check for number of wavs
	# wavs = os.listdir("/z/mkperez/Speech-Data-Raw/HD-SS/segmented")
	# print(len(set([i.split('-')[0] for i in wavs])), len(healthy_all_late.keys()))
	# exit()
	###VARIABLE INIT###
	if len(sys.argv) != 5:
		print("Not enough arguments")
		exit()
	#kaldi_path="/z/mkperez/PremanifestHD-classification/Data/GF-Passage"
	raw_data_path=sys.argv[1]
	kaldi_path=sys.argv[2]
	overwrite=sys.argv[3]
	lexiconPath='/z/mkperez/SpontSpeech-HD/Data/global-dict/lexicon.txt'

	#Set labels based on label type
	label_type = sys.argv[4]
	labels = {}
	if label_type == "manifest":
		labels = early_late_balanced
	elif label_type == "late":
		labels = healthy_all_late

	###create folds using LOO###
	k_SS_path=kaldi_path+'_SS_'+label_type
	create_folds(k_SS_path, len(labels.keys()), overwrite)

	###create kaldi data###
	fold_num=0
	for key, set_num in zip(labels.keys(), range(len(labels.keys()))): #key=participant elem, set_num=fold_num
		# if key != '35063' and key != '47647':
		# 	continue
		fold_num+=1
		print("SS FOLD", fold_num, key)
		val_set=[key]
		train_set=labels.keys()
		train_set.remove(key)
		set_path=k_SS_path+"/set"+str(set_num)

		train_set.sort()
		val_set.sort()

		createWavScp(rawDataPath=raw_data_path, kaldiPath=set_path+'/train', itemSet=train_set)
		createWavScp(rawDataPath=raw_data_path, kaldiPath=set_path+'/val', itemSet=val_set)
		createData(raw_data_path, set_path + '/train', lexiconPath, train_set)
		createData(raw_data_path, set_path + '/val', lexiconPath, val_set)

	# ## Phonation Features:
	# 	###create folds using LOO###
	# k_phone_path=kaldi_path+'_phonation_'+label_type
	# create_folds(k_phone_path, len(labels.keys()), overwrite)

	# ###create kaldi data###
	# fold_num=0
	# for key, set_num in zip(labels.keys(), range(len(labels.keys()))): #key=participant elem, set_num=fold_num
	# 	# if key != '35063' and key != '47647':
	# 	# 	continue
	# 	fold_num+=1
	# 	print("Phonation FOLD", fold_num, key)
	# 	val_set=[key]
	# 	train_set=labels.keys()
	# 	train_set.remove(key)
	# 	set_path=k_phone_path+"/set"+str(set_num)

	# 	train_set.sort()
	# 	val_set.sort()

	# 	createWavScp(rawDataPath=raw_data_path, kaldiPath=set_path+'/train', itemSet=train_set)
	# 	createWavScp(rawDataPath=raw_data_path, kaldiPath=set_path+'/val', itemSet=val_set)
	# 	createData(raw_data_path, set_path + '/train', lexiconPath, train_set)
	# 	createData(raw_data_path, set_path + '/val', lexiconPath, val_set)






if __name__ == '__main__':
	main()
