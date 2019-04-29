Created by Matthew Perez

updatedlabels = {'84596': 1, '78971': 1, '95465': 0, '13691': 0, '82975': 0, '76883': 1, '23789': 0, 
'87083': 0, '71834': 0, '78080': 0, '69695': 1, '68117': 0, '69879': 0, '14630': 1,
'75256': 0, '59826': 1, '11739': 0, '88947': 0, '47647': 0, '25066': 0, '52053': 0,
'16486': 0, '18771': 0,	 '52080': 1, '45758': 0, '01634': 1, '91117': 0, '46352': 0, 
'20221': 1, '26753': 0, '29735': 0, '61496': 0, '69573': 1, '29758': 0,'47939': 0, '58812': 0,
'44209': 0, '07920': 0, '05068': 0}

Pairs-20 = {'46352': 0, '20221': 1, '78971': 1, '52053': 0, '82975': 0, '69573': 1, '23789': 0, '75256': 0, '76883': 1, '29758': 0, '52080': 1, '78080': 0, '69695': 1, '01634': 1, '91117': 0, '59826': 1, '87083': 0, '14630': 1, '84596': 1, '71834': 0}


# Data # 16k sample rate
Choosing healthy speakers with best speaking rating
Premanifest (10 speakers). 07729 doesnt have spontaneous speech transcript (british speakers).  68736 not used due to uncofirmed genetic test
Healthy (29 speakers): 

Fisher-Corpus 8k sample rate

# Kaldi #
Extract just the participant side of conversation

GRANDFATHER PASSAGE
Scripts/kaldi/run.sh for creating kaldi files (note: GMM=2000 has highest performance for GF-AM)
Scripts/run.sh


Open Questions:
10 premanifest, 29 healthy = Choose balanced dataset? Balanced Dataset and use matching age

Extracting Features:
Pronunciation (acoustic) -> Purpose is to identify outlier pronunciation (highly variable phone pronunciation). Ideally can we identify mispronunciations? Challenge with unscripted speech though is that WER is bad -> How can we know what was exactly said without transcriptions? Use GF Passage which we know to have simple and easy word boundaries making forced alignment easy.

Linguistic features? -> Readability index (Chance of getting wide variantions in results of same text) https://www.geeksforgeeks.org/readability-index-pythonnlp/

Conersational (lexical) -> Speech rate and pauses

Entrainment (acoustic)?

18 controls -> adaptation data. Acoustic models

TODO:
-Forced alignment using word boundaries 
-SpeakerID + pre-trained ASR
-Extract above features
-Machine learning using GF-Passage Features, Ensemble features 



Focus on Spontaneous Speech exclusively. 
Extract kaldi features for processing. Apply final.mdl and G.fst from aspire

