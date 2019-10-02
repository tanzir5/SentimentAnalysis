from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import SGD
from gensim.models import Word2Vec
import numpy
import gensim
from keras.utils import to_categorical

dimension = 0
sentiment = {int:float}
mxSize = 5
totalData = 0
X = []
Y = []

def error(message):
	print(message)
	exit(0)

def processDatasetX():
	global dimension, sentiment, mxSize, totalData, X, Y
	
	#word2vecModel = Word2Vec.load("sentimentTreebankWord2vec.model")
	word2vecModel = gensim.models.KeyedVectors.load_word2vec_format("Data/GoogleNews-vectors-negative300.bin", binary=True, limit=100000)

	wordVectors = word2vecModel.wv
	dimension = word2vecModel.vector_size	
	zeroVector = [0] * dimension
	
	
	for line in open("Data/dictionary.txt"):
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
			mxSize = max(mxSize, len(curDataX))
			totalData += 1
	
	for i in range(len(X)):
		st = len(X[i])
		for j in range(st, mxSize):
			X[i].append(zeroVector)
				
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
	
def processDataset():
	processDatasetY()
	processDatasetX()

def verifyDataset():
	global X
	print("---------------------------------")
	for i in X: 
		if(len(i) != mxSize): 
			print(len(i))
			exit(0)
				
processDataset()

print("Max size of X: " + str(mxSize) + ", totalData: " + str(totalData));

verifyDataset()
print("OK")


dataY = numpy.array(Y)
print(dataY.shape)
dataX = numpy.array(X)
print(dataX.shape)
dataX = dataX.reshape(len(X), len(X[0]), len(X[0][0]), 1)
print(dataX.shape)
print(mxSize, dimension, "1")
#dataY = to_categorical(dataY)
#create model
model = Sequential()
model.add(Conv2D(300, kernel_size=(5,dimension), activation="linear", input_shape=(mxSize, dimension, 1)))
model.add(MaxPooling2D(pool_size=(mxSize-4,1)))
model.add(Flatten())
model.add(Dense(300, activation = "sigmoid"))
model.add(Dense(2, activation="softmax"))
sgd = SGD(learning_rate=0.01)
model.compile(optimizer = sgd, loss='binary_crossentropy', metrics=['mse'])
model.fit(dataX, dataY, epochs=15)
model.save("FirstModel.h5")

