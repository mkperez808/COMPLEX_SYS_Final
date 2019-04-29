#Matthew Perez
import os
from io import open
#missing transcripts
#['07729'=premanifest, '40216'=healthy, '42080'=Early, '53870'=Early, '82697'=healthy]

# Unidetified (No GF Passage or transcript): 53228.wav, 76550.wav

#Need to do these transcriptions
#69695 wav and txt dont line up
#68736 = HDA007

#Better Transcript for '84596': 1

# labels = {'40216': 0, '82697': 0, '07729': 1, '84596': 1, '78971': 1, '95465': 0, '13691': 0, '82975': 0, '76883': 1, '23789': 0, 
# '87083': 0, '71834': 0, '78080': 0, '69695': 1, '68117': 0, '69879': 0, '14630': 1,
# '75256': 0, '59826': 1, '11739': 0, '88947': 0, '47647': 0, '25066': 0, '52053': 0,
# '16486': 0, '18771': 0,	 '52080': 1, '45758': 0, '01634': 1, '91117': 0, '46352': 0, 
# '20221': 1, '26753': 0, '29735': 0, '61496': 0, '69573': 1, '29758': 0,'47939': 0, '58812': 0,
# '44209': 0, '07920': 0, '68736': 1, '05068': 0}

# labels = {'82697': 0, '78971': 1, '95465': 0, '13691': 0, '82975': 0, '44574': 3, '35063': 3,
# '23789': 0, '87083': 0, '71834': 0, '76373': 3, '69879': 0, '78080': 0, '69695': 1, '38717': 2,
# '68117': 0, '00359': 2, '81392': 3, '14630': 1, '95407': 3, '07729': 1, '56896': 3, '75256': 0,
# '76883': 1, '24371': 2, '78597': 2, '59826': 1, '11739': 0, '18261': 2, '53870': 2, '84596': 1,
# '61496': 0, '88947': 0, '47647': 0, '32762': 2, '25066': 0, '52053': 0, '16486': 0, '40216': 0,
# '18771': 0, '52080': 1, '45758': 0, '50377': 2, '91117': 0, '07920': 0, '46352': 0, '20221': 1,
# '26753': 0, '69573': 1, '73828': 2, '29735': 0, '29758': 0, '47939': 0, '80292': 3, '55029': 2,
# '58812': 0, '44209': 0, '42080': 2, '68736': 1, '05068': 0, '33752': 2, '01634': 1}


updatedlabels = {'84596': 1, '78971': 1, '95465': 0, '13691': 0, '82975': 0, '76883': 1, '23789': 0, 
'87083': 0, '71834': 0, '78080': 0, '69695': 1, '68117': 0, '69879': 0, '14630': 1,
'75256': 0, '59826': 1, '11739': 0, '88947': 0, '47647': 0, '25066': 0, '52053': 0,
'16486': 0, '18771': 0,	 '52080': 1, '45758': 0, '01634': 1, '91117': 0, '46352': 0, 
'20221': 1, '26753': 0, '29735': 0, '61496': 0, '69573': 1, '29758': 0,'47939': 0, '58812': 0,
'44209': 0, '07920': 0, '05068': 0}

def extract_line(line):
	'''
	Return None if line is invalid (has time or other non-speaking) or empty
	Return stripped line
	'''
	line = line.lower().rstrip()
	line_arr = line.split()
	#If line is empty -> Return None
	if not line_arr:
		return None

	print("line_arr", line_arr)
	first_word = line_arr[0]
	first_word = first_word.split(":")[0]
	print('first_word', first_word)
	if first_word == "p" or first_word == "participant":
		#Participant 
		return line_arr[1:]

	elif first_word == "r" or first_word == "ra":
		#Clinician
		return None
	else:
		return None


def check_wavs(path):
	count=0
	missing_labels = labels.keys()
	files = os.listdir(path)
	#print(files)
	for f in files:
		if f.endswith(".wav"):
			count+=1
			name = f.split(".")[0]
			if name in labels.keys():
				missing_labels.remove(name)
	print('missing wavs', missing_labels)
	print('total wavs ', count)
	exit()


def ling_feats(path):
	count=0
	missing_labels = labels.keys()
	files = os.listdir(path)
	#print(files)
	for f in files:
		if f.endswith(".txt"):
			name = f.split(".")[0]
			file_path = os.path.join(path, f)
			print(file_path)

			with open(file_path, 'r', encoding='utf-8') as r:
				for lines in r:
					print('line', lines, 'file', name)
					if extract_line(lines):
						line = extract_line(lines)
						print('line', line)
						exit()


			exit()


	# 		if name in labels.keys():
	# 			print(name)
	# 			count+=1
	# 			missing_labels.remove(name)
	# print(count)
	# print(missing_labels)
	# exit()
	#with open()






def main():
	SS_speech = '/z/mkperez/Speech-Data-Raw/HD-SS'
	check_wavs(SS_speech)
	ling_feats(SS_speech)
	



if __name__=="__main__":
	main()
