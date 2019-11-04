from keras.models import Sequential
from keras.layers import concatenate, Dense, Conv2D, Flatten, MaxPooling2D, Input
from keras.optimizers import SGD
from keras.models import Model
from gensim.models import Word2Vec
import numpy
import gensim
from keras.utils import to_categorical

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag

homeDir = "/home/robolab/Desktop/Tanzir/SentimentAnalysis/SentimentAnalysis/English_StanfordSentimentTreebank_Word2Vec/"


POSITIVE = [1, 0, 0, 0]
NEGATIVE = [0, 1, 0, 0]
NEUTRAL = [0, 0, 1, 0]
AMBIGUOUS = [0, 0, 0, 1]

ADJ =     [1, 0, 0, 0]
NOUN =    [0, 1, 0, 0]
ADV =     [0, 0, 1, 0]
VERB =    [0, 0, 0, 1]

word2vecModel = gensim.models.KeyedVectors.load_word2vec_format(homeDir+"Data/GoogleNews-vectors-negative300.bin", binary=True)
wordVectors = word2vecModel.wv
dimensionWV = word2vecModel.vector_size
dimensionPosPol = 8	
zeroVectorWV = [0] * dimensionWV
zeroVectorPosPol = [0] * dimensionPosPol

sentiment = {int:float}
mxSize = 60
Xtrain = []
Ytrain = []
Xtest = []
Ytest = []
def error(message):
	print(message)
	exit(0)



#can be modified to be better
def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return (wn.ADJ, ADJ)
    elif tag.startswith('N'):
        return (wn.NOUN, NOUN)
    elif tag.startswith('R'):
        return (wn.ADV, ADV)
    elif tag.startswith('V'):
        return (wn.VERB, VERB)
    
    return (None, None)

def singleDataFormat(data):
	doc = ""
	for w in data:
		doc += w
		doc += " "
	doc = doc[:-1]
	WV = []
	totalWords = 0
	posPol = []
	#print("DATA")
	#print(doc)
	raw_sentences = sent_tokenize(doc)
	#print("OKDATA")
	eps = 0.1
	threshhold = 0.5
	
	for raw_sentence in raw_sentences:
		tagged_sentence = pos_tag(word_tokenize(raw_sentence))

		for word, tag in tagged_sentence:
			(wn_tag, tmp) = penn_to_wn(tag)
			
			if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
				#print(word, wn_tag)
				continue
			lemma = WordNetLemmatizer().lemmatize(word, pos=wn_tag)
			#print(word, lemma)
			if not lemma:
				#print(word, lemma, "2")
				continue
			synsets = wn.synsets(lemma, pos=wn_tag)
			if not synsets:
				#print(word)
				continue
			
			# Take the first sense, the most common
			synset = synsets[0]
			swn_synset = swn.senti_synset(synset.name())
			
			
			if word in wordVectors:
				totalWords += 1
			
	ok = True
	if totalWords >= 4: 
		ok = True
	else: 
		ok = False
		#print(doc)
	return ok
	
def processDatasetX(location, dest):
	global  sentiment, mxSize, totalData, zeroVectorWV
	
	writer = open(dest, "w")
	
	accepted = 0
	rejected = 0
	for line in open(location):
		line = line.lower()
		s = line.split()
		idString = s[-1][s[-1].find("|")+1:]
		lastWord = s[-1][0:s[-1].find("|")]
		id = int(idString)
		#print(s, idString)
		s = s[:-1] 
		s.append(lastWord)
		
		#print(s)
		if(len(s) < 4):
			rejected += 1
			continue
		ok = singleDataFormat(s)
		
		if(ok):
			writer.write(line)
			accepted += 1
		else:
			rejected += 1
	
	print("ac: " + str(accepted) + ", rejected: " + str(rejected))
	writer.close()

def processDataset():
	processDatasetX(homeDir+"Data/train.txt", homeDir+"Data/cleanTrain.txt")
	processDatasetX(homeDir+"Data/test.txt", homeDir+"Data/cleanTest.txt")
	
'''
def verifyDataset():
	global X
	print("---------------------------------")
	for i in X: 
		if(len(i) != mxSize): 
			print(len(i))
			exit(0)
'''

				


print("START")
processDataset()
