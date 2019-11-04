from keras.models import load_model
import numpy
from gensim.models import Word2Vec
import gensim

mxSize = 55
word2VecModel = gensim.models.KeyedVectors.load_word2vec_format("Data/GoogleNews-vectors-negative300.bin", binary=True, limit=100000)
wordVectors = word2VecModel.wv
dimension = word2VecModel.vector_size
zeroVector = [0] * dimension
model = load_model("FirstModel.h5")

def vectorizeSentence(line):
	global wordVectors, zeroVector
	curDataX = []
	for word in line.split(): 
		if word in wordVectors:
			curDataX.append(wordVectors[word])
		else: 
			curDataX.append(zeroVector)
	for i in range(len(curDataX), mxSize): 
		curDataX.append(zeroVector)
	return curDataX

def solve(line): 
	line = line.lower()
	X = vectorizeSentence(line)
	dataX = numpy.array(X)
	dataX = dataX.reshape(1, mxSize, dimension, 1)
	ret = model.predict(dataX)
	print(ret)
	if(ret[0][0] < ret[0][1]):
		return "Negative"
	else: 
		return "Positive"

model.summary()
	

while (True): 
	line = input("Enter the line: ")
	if(line == "EXIT"):
		break
	res = solve(line)
	print("The sentiment is: " + res)

print("Good bye")
