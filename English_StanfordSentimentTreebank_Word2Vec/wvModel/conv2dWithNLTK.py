from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import SGD
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


dimension = 300
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
			
			if word in wordVectors:
				totalWords += 1
				WV.append(wordVectors[word])
			else: 
				WV.append(zeroVectorWV)		
			
	ok = True
	if totalWords >= 4: 
		ok = True
	else: 
		ok = False
		#print(doc)
	return (WV, ok)
	
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
		(wvCur, ok) = singleDataFormat(s)
		
		if(ok):
			wvX.append(wvCur)
			Y.append([sentiment[id], 1-sentiment[id]])
			accepted += 1
		else: 
	#		print(line)
			rejected += 1
	
	for i in range(len(wvX)):
		st = len(wvX[i])
		for j in range(st, mxSize):
			wvX[i].append(zeroVectorWV)
			
	print("ac: " + str(accepted) + ", rejected: " + str(rejected))
	return (wvX,  Y)
				
def processDatasetY():
	positive = 0
	negative = 0
	start = True
	for line in open(homeDir + "Data/sentiment_labels.txt"):
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

def processDataset():
	global Xtrain, Ytrain, Xtest, Ytest
	processDatasetY()
	(Xtrain, Ytrain) = processDatasetX(homeDir + "Data/cleanTrain.txt")
	(Xtest, Ytest) = processDatasetX(homeDir + "Data/cleanTest.txt")
	
'''
def verifyDataset():
	global X
	print("---------------------------------")
	for i in X: 
		if(len(i) != mxSize): 
			print(len(i))
			exit(0)
'''


def binaryMetric(y_true, y_pred):
	correct = 0
	for i in range(0, y_true.shape[0]): 
		if( (y_true[i][0] <= 0.5 and y_pred[i][0] <= 0.5) or (y_true[i][0] >= 0.5 and y_pred[i][0] >= 0.5)) : 
			correct += 1
	return (correct * 100.0)/ed


def configureDataset(X,Y):
	dataY = numpy.array(Y)
	dataX = numpy.array(X)
	dataX = dataX.reshape(len(X), len(X[0]), len(X[0][0]), 1)
	return (dataX, dataY)
				
processDataset()

#print("Max size of X: " + str(mxSize) + ", totalData: " + str(totalData));




(trainX, trainY) = configureDataset(Xtrain, Ytrain)
(testX, testY) = configureDataset(Xtest, Ytest)

model = Sequential()
model.add(Conv2D(300, kernel_size=(5,dimension), activation="linear", input_shape=(mxSize, dimension, 1)))
model.add(MaxPooling2D(pool_size=(mxSize-4,1)))
model.add(Flatten())
model.add(Dense(300, activation = "sigmoid"))
model.add(Dense(2, activation="softmax"))
sgd = SGD(learning_rate=0.01)
model.compile(optimizer = sgd, loss='binary_crossentropy', metrics=['mse'])
model.fit(trainX, trainY, validation_data = (testX, testY), epochs=100)
model.save("wvModelCleanWithNLTK_1.h5")

