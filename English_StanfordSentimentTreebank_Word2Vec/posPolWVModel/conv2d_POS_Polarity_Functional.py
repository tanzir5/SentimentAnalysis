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

ADJ =     [1, 0, 0, 0, 0]
NOUN =    [0, 1, 0, 0, 0]
ADV =     [0, 0, 1, 0, 0]
VERB =    [0, 0, 0, 1, 0]
OTHERS =  [0, 0, 0, 0, 1]

word2vecModel = gensim.models.KeyedVectors.load_word2vec_format(homeDir+"Data/GoogleNews-vectors-negative300.bin", binary=True)
wordVectors = word2vecModel.wv
dimensionWV = word2vecModel.vector_size
dimensionPosPol = 9	
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
    
    return (None, OTHERS)

def singleDataFormat(data):
	global POSITIVE, NEGATIVE, NEUTRAL, AMBIGUOUS, wordVectors
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
			posPolCur = tmp.copy()
			
			if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
				posPolCur += NEUTRAL
			else:
				lemma = WordNetLemmatizer().lemmatize(word, pos=wn_tag)
				#print(word, lemma)
				if not lemma:
					print(word, lemma, "2")
					error("NOT LEMMA")
					exit(0)
					
				synsets = wn.synsets(lemma, pos=wn_tag)
				if not synsets:
					posPolCur += NEUTRAL
				else:
					# Take the first sense, the most common
					synset = synsets[0]
					swn_synset = swn.senti_synset(synset.name())
					
					if swn_synset.obj_score() >= threshhold:
						posPolCur += NEUTRAL				
					elif swn_synset.pos_score() > swn_synset.neg_score() + eps: 
						posPolCur += POSITIVE
					elif swn_synset.neg_score() > swn_synset.pos_score() + eps:
						posPolCur += NEGATIVE
					else: 
						posPolCur += AMBIGUOUS
				
			if word in wordVectors:
				totalWords += 1
				WV.append(wordVectors[word])
			else: 
				WV.append(zeroVectorWV)		
			posPol.append(posPolCur)	
			
	ok = True
	if totalWords >= 4: 
		ok = True
	else: 
		ok = False
		#print(doc)
	return (WV, posPol, ok)
	
def processDatasetX(location):
	global  sentiment, mxSize, totalData, zeroVectorWV
	
	wvX = []
	posPolX = []
	Y = []

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
		(wvCur, posPolCur, ok) = singleDataFormat(s)
		
		if(ok):
			wvX.append(wvCur)
			posPolX.append(posPolCur)
			Y.append([sentiment[id], 1-sentiment[id]])
			accepted += 1
		else: 
	#		print(line)
			rejected += 1
	
	for i in range(len(wvX)):
		st = len(wvX[i])
		for j in range(st, mxSize):
			wvX[i].append(zeroVectorWV)
			posPolX[i].append(zeroVectorPosPol)
	
	print("ac: " + str(accepted) + ", rejected: " + str(rejected))
	return (wvX, posPolX, Y)
				
def processDatasetY():
	positive = 0
	negative = 0
	start = True
	for line in open(homeDir+"Data/sentiment_labels.txt"):
		if(start): 
			start = False
		else:
			data = line.split("|")
			if(len(data) != 2):
				error("Inside processDatasetY(), there are not two datas.")
			id = int(data[0])
			label = float(data[1])
			sentiment[id] = label
			'''
			if(label >= 0.5): 
				sentiment[id] = 1
				positive += 1
			else: 
				sentiment[id] = 0
				negative += 1
			
	print(positive, negative)
	'''


def toNumpy2(X,Y):
	dataY = numpy.array(Y)
	dataX = numpy.array(X)
	print(len(X))
	print(len(X[0]))
	print(len(X[0][0]))
	print(type(X[0][0]))
	dataX = dataX.reshape(len(X), len(X[0]), len(X[0][0]), 1)
	return (dataX, dataY)

def toNumpy1(X):
	dataX = numpy.array(X)
	print(len(X))
	print(len(X[0]))
	print(len(X[0][0]))
	print(type(X[0][0]))
	dataX = dataX.reshape(len(X), len(X[0]), len(X[0][0]), 1)
	return dataX
	
def processDataset():
	processDatasetY()
	(tempWVX, tempPPX, tempY) = processDatasetX(homeDir+"Data/cleanTrain.txt")
	(xTrain, yTrain) = toNumpy2(tempWVX, tempY) 
	trainPosPol = toNumpy1(tempPPX)
	(tempWVX, tempPPX, tempY) = processDatasetX(homeDir+"Data/cleanTest.txt")
	(xTest, yTest) =  toNumpy2(tempWVX, tempY)
	testPosPol = toNumpy1(tempPPX)
	return (xTrain, trainPosPol, yTrain, xTest, testPosPol, yTest)
	
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
#wordVectorLayers
#print("Max size of X: " + str(mxSize) + ", totalData: " + str(totalData));
(trainX_WV, trainX_PosPol, trainY, testX_WV, testX_PosPol, testY) = processDataset()
print("OK")

inputWV = Input(shape = (mxSize, dimensionWV, 1))
convWV = Conv2D(300, kernel_size=(5,dimensionWV), activation="linear")(inputWV)
maxPoolingWV = MaxPooling2D(pool_size=(mxSize-4,1))(convWV)
flattenWV = Flatten()(maxPoolingWV)


inputPP = Input(shape = (mxSize, dimensionPosPol, 1))
convPP = Conv2D(300, kernel_size=(5,dimensionPosPol), activation="linear")(inputPP)
maxPoolingPP = MaxPooling2D(pool_size=(mxSize-4,1))(convPP)
flattenPP = Flatten()(maxPoolingPP) 

merged = concatenate([flattenWV, flattenPP])


dense = Dense(300, activation = "sigmoid")(merged) 
output = Dense(2, activation="softmax")(dense) 





print("OKOK")
sgd = SGD(learning_rate=0.01)
model = Model(inputs = [inputWV, inputPP], outputs = output)
model.compile(optimizer = sgd, loss='binary_crossentropy', metrics=['mse'])
model.fit([trainX_WV, trainX_PosPol], trainY, validation_data = ([testX_WV, testX_PosPol], testY), epochs=100)
model.save("FunctionalPosPolWV_2.h5")
print("DONE")

  
