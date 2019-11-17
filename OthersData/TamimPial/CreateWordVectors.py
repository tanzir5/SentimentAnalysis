
# coding: utf-8

# In[71]:

import pickle as cp
import pandas as pd
import gensim
import logging
import random
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[9]:

alltxt = cp.load(open('sachalCorpus.pkl','rb'))


# In[10]:

sentences = []


# In[11]:

for txt in alltxt:
    curSentences = alltxt[0].split('।')
    for sentence in curSentences:
        sentences.append(sentence.split(' '))


# In[12]:

def readComments(f_name):
    comments = []
    f_in = open(f_name)
    while True:
        s = f_in.readline()
        if s == '':
            break
        else:
            comments.append(s.strip().split(' '))
    print(comments[0])
    return comments


# In[13]:

sentences += readComments('cleanedCommentsProthomAlo.txt')
sentences += readComments('cleanedCommentsSachal.txt')


# In[66]:

vectorLen = 100


# In[17]:

model = gensim.models.Word2Vec(sentences=sentences, size=vectorLen, min_count=1, window= 5, workers=4)


# In[18]:

model.most_similar('পুরুষ')


# In[21]:

model.most_similar('খেলা')


# In[23]:

model.most_similar('সুন্দর')


# In[32]:

model.most_similar('খারাপ')


# In[34]:

model.doesnt_match("মানুষ গরু ছাগল মহিষ".split())


# In[35]:

model.most_similar('সরকার')


# In[36]:

df = pd.read_csv('AnnotatedComments.csv')


# In[37]:

def findMaxOccur(id,df):
    cntDict = {'positive':0,'negative':0}
    n_row = {'id':id}
    for index,row in df.iterrows():
        cntDict[row['annotation']]+=1
        n_row['comment'] = row['comment']
        
    if cntDict['negative']>cntDict['positive']:
        n_row['annotation'] = 'negative'
    else:
        n_row['annotation'] = 'positive'
    return n_row
    
    
    

def mergeDuplicates(original_df):
    grouped = original_df.groupby('id')
    mergedDf = pd.DataFrame(columns=['comment','annotation'])
    for id,df in grouped:
        if(len(df) ==1):
            mergedDf = mergedDf.append(df,ignore_index=True)
        else:
            
            mergedDf = mergedDf.append(findMaxOccur(id,df),ignore_index=True)

    return mergedDf[['comment','annotation']]


# In[38]:

df = mergeDuplicates(df)


# In[41]:

df['annotation'] = df['annotation'].apply(lambda sentiment: 1 if sentiment=='positive' else 0 )


# In[53]:

def createTrainTest(df,splitRatio):
    random.seed(24)
    df = df.sample(frac=1)
    idx = int(splitRatio*len(df))
    upperHalf = df[0:idx]
    lowerHalf = df[idx:]
    X_train = upperHalf['comment']
    X_test = lowerHalf['comment']
    y_train = upperHalf['annotation']
    y_test = lowerHalf['annotation']
    
    
    return X_train,X_test,y_train,y_test
    
    
    


# In[96]:

X_train,X_test,y_train,y_test = createTrainTest(df,.9)


# In[115]:

def padZeroRow(model,comment,maxLen,vectorLen):
    print('heu')
    row = []
    commentList = comment.split(' ')
    minIdx = min(len(commentList),maxLen)
    maxIdx = max(len(commentList),maxLen)
    print(commentList)
    for i in range(0,minIdx):
        print(i)
        break
        row.append(model[commentList[i]])
    
    for i in range(minIdx,maxIdx):
        row.append(np.zeros(vectorLen))
    return row
    
    
    

def padZeros(model,X,maxLen,vectorLen):
    
    newX = []
    for comment in X:
        row = padZeroRow(model,comment,maxLen,vectorLen)
        print(row)
        newX.append(row)
        break
        
    return newX
    
    
    
    
    


# In[116]:

stepSize = 20
X_train = padZeros(model,X_train,stepSize,vectorLen)
X_test =  padZeros(model,X_test,stepSize,vectorLen)


# In[89]:

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence


# In[90]:

model = Sequential()


# In[92]:

model.add(LSTM(100,input_shape =(stepSize,vectorLen)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[93]:

model.fit(X_train, y_train, nb_epoch=3, batch_size=64)


# In[ ]:



