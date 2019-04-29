#Matthew Perez
#Generate bigram text
import sys

def main():
	if len(sys.argv) !=3:
		print('error, incorrect number of arguments')
		exit()
	
	train_text = sys.argv[1]
	LM_file = sys.argv[2]

	#Go through train_text
	with open(LM_file, 'w') as w:
		with open(train_text, 'r') as r:
			for line in r:
				#line = line.rstrip()
				new_line_arr = line.split(' ')[1:] #ignore utt-id


				new_line = ' '.join(new_line_arr)

				w.write(new_line)






if __name__=="__main__":
	main()