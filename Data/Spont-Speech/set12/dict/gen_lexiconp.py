import re
with open("lexiconp.txt", "w") as f_w:
	with open("lexicon.txt", "r") as f_r:
		content = f_r.readlines()

		for line in content:
			line_list = re.split('\s+|\t', line)
			str_builder = line_list[0]+'\t1.0\t'+' '.join(line_list[1:])+'\n'
			#print(str_builder)
			f_w.write(str_builder)
