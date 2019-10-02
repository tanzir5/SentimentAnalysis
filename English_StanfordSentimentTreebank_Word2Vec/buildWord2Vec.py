import gensim
import os
import time

cntSentence = 0
cntWord = 0
class MySentences(object):

	def __init__(self, dirName):
		self.dirName = dirName
		
	def __iter__(self):
		global cntSentence, cntWord
		for line in open("Data/original_rt_snippets.txt"):
			line = line.lower()
			sentences = line.split(".")
			cntSentence += len(sentences)
			for s in sentences: 
				if len(s.split()) > 3:
					cntWord += len(s.split())
					yield s.split()


             
sentences = MySentences("AllData")

print("Starting training")
st = time.time()
model = gensim.models.Word2Vec(sentences, size = 120, window = 5, workers=4, min_count = 5)
ed = time.time()

print("Time Taken:")
print(ed-st)
print(cntSentence)
print(cntWord)
model.save("sentimentTreebankWord2vec.model")
print("ALL OK")
