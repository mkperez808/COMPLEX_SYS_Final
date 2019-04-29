import os
import re
import pandas as pd
import time
import numpy as np
import sys





def main():
	if len(sys.argv) !=3:
		print("not enough args")
		exit()

	labels = {'82697': 0, '78971': 1, '95465': 0, '13691': 0, '82975': 0, '76883': 1, '23789': 0,
'87083': 0, '71834': 0, '78080': 0, '69695': 1, '68117': 0, '69879': 0, '14630': 1, '07729': 1,
'75256': 0, '59826': 1, '11739': 0, '84596': 1, '88947': 0, '47647': 0, '25066': 0, '52053': 0,
'16486': 0, '40216': 0, '18771': 0, '52080': 1, '45758': 0, '01634': 1, '91117': 0, '46352': 0,
'20221': 1, '26753': 0, '29735': 0, '61496': 0, '69573': 1, '29758': 0, '47939': 0, '58812': 0,
'44209': 0, '07920': 0, '68736': 1, '05068': 0}

	data_path = sys.argv[1]
	AM = sys.argv[2]

	HC_wer={} #{HD/HC:[wer, ins, del, sub]}
	HD_wer={}

	wer_all=[]
	ins_all=[]
	del_all=[]
	sub_all=[]
	tot_wer={}


	for i in sorted(os.listdir(data_path)):
		#check if HD or HC val speaker
		print('set: ', i)
		set_path = data_path + '/' + i
		spkr_path = set_path +'/val/spk2utt'
		with open(spkr_path, 'r') as r:
			first_line = r.readline()
			spkr = first_line.split(' ')[0][3:] #get speaker
			# print('first_line', first_line)
			# print('spkr ', spkr)
			# exit()


			#for acoustic_model, index in zip(['/exp', '/exp_HC', '/exp_HD'], [1,2,3]):
			best_wer_file = set_path + '/exp/'+AM+'/decode_val/scoring_kaldi/best_wer'
			if not os.path.isfile(best_wer_file):
				print 'file \'%s\' does not exist' %best_wer_file
				exit()
				#continue

			with open(best_wer_file) as logfile: #get WER
				for line in logfile.readlines():
					if line.startswith("%WER"):
						# print(line)
						# exit()
						m = re.match(r'\%WER\s*([0-9]+)\.([0-9]+)\s*\[\s*([0-9]+)\s*\/\s*([0-9]+),\s*([0-9]+)\s*ins,\s*([0-9]+)\s*del,\s*([0-9]+)\s*sub\s*\]', line)
						
						#print('len', len(m.group(1)))
						if len(m.group(1))==1:
							dec = float('0.0'+m.group(1)+m.group(2))
						elif len(m.group(1))==2:
							dec = float('0.'+m.group(1)+m.group(2))
						else:
							#What is here?
							print(m.group(1))
							exit()

						if labels[spkr] == 0: #HC
							if len(HC_wer.keys())==0:
								HC_wer['wer']=[]
								HC_wer['ins']=[]
								HC_wer['del']=[]
								HC_wer['sub']=[]

							HC_wer['wer'].append(dec)
							HC_wer['ins'].append(int(m.group(5)))
							HC_wer['del'].append(int(m.group(6)))
							HC_wer['sub'].append(int(m.group(7)))
						else: #HD
							if len(HD_wer.keys())==0:
									HD_wer['wer']=[]
									HD_wer['ins']=[]
									HD_wer['del']=[]
									HD_wer['sub']=[]
							HD_wer['wer'].append(dec)
							HD_wer['ins'].append(int(m.group(5)))
							HD_wer['del'].append(int(m.group(6)))
							HD_wer['sub'].append(int(m.group(7)))


	#Compute mean and std for both HD and HC
	tot_wer['wer_mean']=np.mean(HD_wer['wer']+HC_wer['wer'])
	tot_wer['wer_std']=np.std(HD_wer['wer']+HC_wer['wer'])
	tot_wer['ins_mean']=np.mean(HD_wer['ins']+HC_wer['ins'])
	tot_wer['ins_std']=np.std(HD_wer['ins']+HC_wer['ins'])
	tot_wer['del_mean']=np.mean(HD_wer['del']+HC_wer['del'])
	tot_wer['del_std']=np.std(HD_wer['del']+HC_wer['del'])
	tot_wer['sub_mean']=np.mean(HD_wer['sub']+HC_wer['sub'])
	tot_wer['sub_std']=np.std(HD_wer['sub']+HC_wer['sub'])

	#HC
	tot_wer['HC_wer_mean']=np.mean(HC_wer['wer'])
	tot_wer['HC_wer_std']=np.std(HC_wer['wer'])
	tot_wer['HC_ins_mean']=np.mean(HC_wer['ins'])
	tot_wer['HC_ins_std']=np.std(HC_wer['ins'])
	tot_wer['HC_del_mean']=np.mean(HC_wer['del'])
	tot_wer['HC_del_std']=np.std(HC_wer['del'])
	tot_wer['HC_sub_mean']=np.mean(HC_wer['sub'])
	tot_wer['HC_sub_std']=np.std(HC_wer['sub'])	
	
	#HD
	tot_wer['HD_wer_mean']=np.mean(HD_wer['wer'])
	tot_wer['HD_wer_std']=np.std(HD_wer['wer'])
	tot_wer['HD_ins_mean']=np.mean(HD_wer['ins'])
	tot_wer['HD_ins_std']=np.std(HD_wer['ins'])
	tot_wer['HD_del_mean']=np.mean(HD_wer['del'])
	tot_wer['HD_del_std']=np.std(HD_wer['del'])
	tot_wer['HD_sub_mean']=np.mean(HD_wer['sub'])
	tot_wer['HD_sub_std']=np.std(HD_wer['sub'])		

	write_path=data_path + '/set0/exp/'+AM+'/wer.txt'
	print('write_path', write_path)
	with open(write_path, 'w') as wer_w:
		for key in sorted(tot_wer.keys()):
			print(key, tot_wer[key])
			wer_w.write(key+" %.3f\n"%tot_wer[key])




if __name__=="__main__":
	main()





