DICT_PATH='/z/mkperez/PremanifestHD/global-dict'
GF_kaldi='/z/mkperez/PremanifestHD-classification/Data/GF-Passage'

#echo $DATA_ROOT $DICT_PATH
#exit
for dir in $GF_kaldi/set*
do
	cp -r $DICT_PATH $dir/dict
done