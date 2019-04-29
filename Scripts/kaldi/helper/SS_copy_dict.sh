DICT_PATH='/z/mkperez/PremanifestHD-classification/Data/global-dict' #Augmented with a few 
SS='/z/mkperez/PremanifestHD-classification/Data/Spont-Speech'
Aspire_dict='/z/mkperez/PremanifestHD-classification/kaldi-model/lang_pp_test'
Swbd_dict='/z/mkperez/kaldi'

#echo $DATA_ROOT $DICT_PATH
#exit
for dir in $SS/set*
do
	# Copy basic dictionary dir
	cp -r $DICT_PATH $dir/dict
done

cp -r $DICT_PATH $SS/all_data_test/test/dict