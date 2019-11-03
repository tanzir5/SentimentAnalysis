from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
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



POSITIVE = [1, 0, 0, 0]
NEGATIVE = [0, 1, 0, 0]
NEUTRAL = [0, 0, 1, 0]
AMBIGUOUS = [0, 0, 0, 1]

ADJ =     [1, 0, 0, 0]
NOUN =    [0, 1, 0, 0]
ADV =     [0, 0, 1, 0]
VERB =    [0, 0, 0, 1]

word2vecModel = gensim.models.KeyedVectors.load_word2vec_format("Data/GoogleNews-vectors-negative300.bin", binary=True)
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
			(wn_tag, posPolCur) = penn_to_wn(tag)
			if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
				continue
 
			lemma = WordNetLemmatizer().lemmatize(word, pos=wn_tag)
			#print(word, lemma)
			if not lemma:
				continue
			synsets = wn.synsets(lemma, pos=wn_tag)
			if not synsets:
				continue
			
			# Take the first sense, the most common
			synset = synsets[0]
			swn_synset = swn.senti_synset(synset.name())
			
			if swn_synset.obj_score() >= threshhold:
				posPol += NEUTRAL				
			elif swn_synset.pos_score() > swn_synset.neg_score() + eps: 
				posPol += POSITIVE
			elif swn_synset.neg_score() > swn_synset.pos_score() + eps:
				posPol += NEGATIVE
			else: 
				posPol += AMBIGUOUS
			
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
	return (WV, posPol, ok)
	
def processDatasetX(location):
	global  sentiment, mxSize, totalData, zeroVectorWV
	
	wvX = []
	posPolX = []
	Y = []

	cnt = 0
	for line in open(location):
		line = line.lower()
		s = line.split()
		idString = s[-1][s[-1].find("|")+1:]
		id = int(idString)
		#print(s, idString)
		s = s[:-1]
		#print(s)
		if(len(s) < 4):
			continue
		(wvCur, posPolCur, ok) = singleDataFormat(s)
		
		if(ok):
			wvX.append(wvCur)
			posPolX.append(posPolCur)
			Y.append([sentiment[id], 1-sentiment[id]])
			cnt += 1
		
	for i in range(len(wvX)):
		st = len(wvX[i])
		for j in range(st, mxSize):
			wvX[i].append(zeroVectorWV)
			posPolX.append(zeroVectorPosPol)
	return (wvX, Y)
				
def processDatasetY():
	positive = 0
	negative = 0
	start = True
	for line in open("Data/sentiment_labels.txt"):
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


def toNumpy(X,Y):
	dataY = numpy.array(Y)
	dataX = numpy.array(X)
	dataX = dataX.reshape(len(X), len(X[0]), len(X[0][0]), 1)
	return (dataX, dataY)

def processDataset():
	processDatasetY()
	(tempX, tempY) = processDatasetX("Data/train.txt")
	(xTrain, yTrain) = toNumpy(tempX, tempY) 
	(tempX, tempY) = processDatasetX("Data/test.txt")
	(xTest, yTest) =  toNumpy(tempX, tempY)
	return (xTrain, yTrain, xTest, yTest)
	
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
(trainX_WV, trainY, testX_WV, testY) = processDataset()
print("OK")
inputWV = Input(shape = (mxSize, dimensionWV, 1))
convWV = Conv2D(300, kernel_size=(5,dimensionWV), activation="linear")(inputWV)
maxPoolingWV = MaxPooling2D(pool_size=(mxSize-4,1))(convWV)
flattenWV = Flatten()(maxPoolingWV)
denseWV = Dense(300, activation = "sigmoid")(flattenWV) 
outputWV = Dense(2, activation="softmax")(denseWV) 

print("OKOK")
sgd = SGD(learning_rate=0.01)
model = Model(inputs = inputWV, outputs = outputWV)
model.compile(optimizer = sgd, loss='binary_crossentropy', metrics=['mse'])
model.fit(trainX_WV, trainY, validation_data = (testX_WV, testY), epochs=15)
model.save("FunctionalModel2.h5")
print("DONE")

####################################################################################################################

'''
  
def swn_polarity(text):
	global POSITIVE, NEGATIVE, NEUTRAL, AMBIGUOUS
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """
 
    text = clean_text(text)
 
	ret = []
 
    raw_sentences = sent_tokenize(text)
    eps = 0.1
    threshhold = 0.5
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
 
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
 
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
 
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
 
            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
			if swn_synset.obj_score() >= threshhold:
                ret.append(NEUTRAL)				
			elif swn_synset.pos_score() > swn_synset.neg_score() + eps: 
                ret.append(POSITIVE)
            elif swn_synset.neg_score() > swn_synset.pos_score() + eps:
                ret.append(NEGATIVE)
            else: 
                ret.append(AMBIGUOUS)				
            
   
    return ret


def configureDatasetPosPol(address):
	for line in open(address):
		line = line.lower()
		
		s = line.split()
		idString = s[-1][s[-1].find("|")+1:]
		id = int(idString)
		s = s[:-1]
		wordInSentence = 0
		curDataX = []
		for word in s: 
			if word in wordVectors:
				wordInSentence += 1
				curDataX.append(wordVectors[word])
			else: 
				curDataX.append(zeroVector)
		if(wordInSentence >= 4):
			X.append(curDataX)
			Y.append([sentiment[id], 1-sentiment[id]])
			
	for i in range(len(X)):
		st = len(X[i])
		for j in range(st, mxSize):
			X[i].append(zeroVector)
	return (X, Y)

#POS_Polarity



trainX_PosPol = configureDatasetPosPol("Data/train.txt")
testX_PosPol = configureDatasetPosPol("Data/test.txt")

'''


