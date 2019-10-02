from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from gensim.models import Word2Vec
import numpy

X = [1, 2, 3, 0, 0, 2]
dataX = numpy.array(X)
dataX.reshape(len(X), 1)
print(dataX.shape)
