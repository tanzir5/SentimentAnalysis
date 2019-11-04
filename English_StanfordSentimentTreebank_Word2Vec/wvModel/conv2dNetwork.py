from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import SGD
from gensim.models import Word2Vec
import numpy
import gensim
from keras.utils import to_categorical

homeDir = "/home/robolab/Desktop/Tanzir/SentimentAnalysis/SentimentAnalysis/English_StanfordSentimentTreebank_Word2Vec/"

dimension = 0
sentiment = {int:float}
mxSize = 60
Xtrain = []
Ytrain = []
Xtest = []
Ytest = []
def error(message):
	print(message)
	exit(0)

def processDatasetX(location):
	global dimension, sentiment, mxSize, totalData
	
	X = []
	Y = []
	
	#word2vecModel = Word2Vec.load("sentimentTreebankWord2vec.model")
	word2vecModel = gensim.models.KeyedVectors.load_word2vec_format(homeDir + "Data/GoogleNews-vectors-negative300.bin", binary=True)

	wordVectors = word2vecModel.wv
	dimension = word2vecModel.vector_size	
	zeroVector = [0] * dimension
	
	
	for line in open(location):
		line = line.lower()
		s = line.split()
		idString = s[-1][s[-1].find("|")+1:]
		lastWord = s[-1][0:s[-1].find("|")]
		id = int(idString)
		#print(s, idString)
		s = s[:-1] 
		s.append(lastWord)
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
model.save("wvModelCleanNoNLTK_1.h5")

