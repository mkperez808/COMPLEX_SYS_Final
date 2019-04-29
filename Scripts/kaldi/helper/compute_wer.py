#Matthew Perez
#Compute average WER for each fold

import sys
import re

#20 healthy/hd pairs
labels = {'46352': 0, '20221': 1, '78971': 1, '52053': 0, '82975': 0, '69573': 1, 
'23789': 0, '75256': 0, '76883': 1, '29758': 0, '52080': 1, '78080': 0, '69695': 1, '01634': 1, 
'91117': 0, '59826': 1, '87083': 0, '14630': 1, '84596': 1, '71834': 0}

fold_path = sys.argv[1]
exp_folder = sys.argv[2]
output_path = sys.argv[3]
num_0 = 0
den_0=0
ins_0=0
dele_0=0
sub_0=0

num_1 = 0
den_1=0
ins_1=0
dele_1=0
sub_1=0

for i in range(0,20):
	spkr_label=0
	#fold = fold_path+'/all_data_test/test/exp/mono_all_ali/decode/scoring_kaldi/best_wer' #For all_Data_path
	text = fold_path+'/set'+str(i)+'/val/text'
	with open(text, 'r') as t_r:
		line = t_r.readline()
		participant = line.split('-')[0]
		intsre = re.findall(r'\d+',participant)[0]
		#print('line', intsre)
		spkr_label = labels[intsre]

	fold = fold_path+'/set'+str(i)+'/exp/'+exp_folder+'/decode/scoring_kaldi/best_wer'
	with open(fold, 'r') as r:
		for line in r:
			#Ensure its the WER line
			if line.startswith('%WER'):
				line_arr = line.split()
				# print(line_arr)
				# exit()
				if spkr_label == 0:
					num_0 += int(line_arr[3])
					den_0 += int(line_arr[5].replace(',','')) #strip end comma
					ins_0 += int(line_arr[6])
					dele_0 += int(line_arr[8])
					sub_0 += int(line_arr[10])
				else:
					num_1 += int(line_arr[3])
					den_1 += int(line_arr[5].replace(',','')) #strip end comma
					ins_1 += int(line_arr[6])
					dele_1 += int(line_arr[8])
					sub_1 += int(line_arr[10])

print('WER total {} frac: {} / {}'.format(exp_folder,num_0+num_1, den_0+den_1))
print('WER total: {}'.format(float(num_0+num_1)/float(den_0+den_1)))
# print('WER HC frac: {} / {}'.format(num_0, den_0))
# print('HC WER: {}'.format(float(num_0)/float(den_0)))
# print('WER HD frac: {} / {}'.format(num_1, den_1))
# print('HD WER: {}'.format(float(num_1)/float(den_1)))

with open(output_path, 'w') as w:
	w.write('WER total frac: {} / {}'.format(num_0+num_1, den_0+den_1))
	w.write('WER total: {}'.format(float(num_0+num_1)/float(den_0+den_1)))

	w.write('WER HC frac: {} / {}'.format(num_0, den_0))
	w.write('HC WER: {}'.format(float(num_0)/float(den_0)))
	w.write('WER HD frac: {} / {}'.format(num_1, den_1))
	w.write('HD WER: {}'.format(float(num_1)/float(den_1)))


