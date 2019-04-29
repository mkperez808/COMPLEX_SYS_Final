
bad_lst=[]

def check_files(f1, f2, t, l):
	with open(f1, 'r') as f_1:
		with open(f2, 'r') as f_2:
			lst_utts = [x.split()[0] for x in f_1]
			# print('list', lst_utts)
			# exit()
			for i in f_2:
				#print(i.split()[0], j.split()[0])

				if i.split()[0] not in lst_utts and i.split()[0] not in bad_lst:
					print('bad', t, l, i.split()[0])
					bad_lst.append(i.split()[0])
					#exit()
	#exit()


if __name__=="__main__":
	Data = '/z/mkperez/PremanifestHD-classification/Data/Spont-Speech'

	for t in ['train', 'val']:
		print('t', t)
		for i in range(20):
			path = Data+'/set'+str(i)+'/'+t

			check_files(path+'/feats.scp', path+'/utt2spk', t, i)