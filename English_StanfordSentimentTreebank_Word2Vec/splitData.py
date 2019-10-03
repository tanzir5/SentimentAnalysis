import random
test = open("Data/test.txt","a+")
train = open("Data/train.txt", "a+")
cntA = 0
cntB = 0
for line in open("Data/dictionary.txt"):
	if len(line) > 3 and random.randint(0, 9) == 4 :
		cntA += 1
		test.write(line)
	else: 
		cntB += 1
		train.write(line)
test.close()
train.close()
print(cntA, cntB)
print(cntA+cntB, cntA/(cntA+cntB)*100.0, cntB/(cntA+cntB)*100.0)
