import sys
import json
import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from scipy import spatial

from keras.preprocessing.text import text_to_word_sequence, one_hot
from keras.models import Sequential
from keras.layers import Dense

def clean(text):
	return text_to_word_sequence(text,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        split=" ")

def seq_2_matrix(sequence, embedding_map):
	m = []
	for word in sequence:
		emb = embedding_map.get(word)
		if emb is not None:
			m.append(emb)
	return np.array(m)

def matrix_2_avg(emb_matrix):
	return np.mean(emb_matrix, 0)

def display_wordvecs(wordvecs):
	maxv = np.max(wordvecs)
	plt.imshow(np.array(wordvecs) / maxv, cmap='RdBu', interpolation='nearest')
	plt.show()

def display_wordvecs_similarity(wordvecs):
	d = [[spatial.distance.cosine(a,b) for a in wordvecs] for b in wordvecs]
	plt.imshow(d, cmap='hot', interpolation='nearest')
	plt.show()

def initEmbeddingMap(fileName='embeddingDict.p'):
	try:
		print('loading saved embedding dictionary')
		embedding_map = pickle.load(open(fileName,'rb'))
	except (OSError, IOError) as e:
		print('constructing embedding dictionary')
		embedding_map = {}
		with open('glove.6B.50d.txt') as glove:
			for line in glove:
			    values = line.split()
			    word = values[0]
			    value = np.asarray(values[1:], dtype='float32')
			    embedding_map[word] = value
		pickle.dump(embedding_map, open(fileName,'wb'))
	return embedding_map

def initRawData(input_file, maxlines = sys.maxsize, fileName='rawData.p', save=True):
	try:
		rawInputData, rawOutputData = pickle.load(open(fileName,'rb'))
		print('loaded saved raw data')
	except (OSError, IOError) as e:
		print('initializing raw data')
		rawInputData = []
		rawOutputData = []
		with open(input_file,'r') as f:
			for i in range(maxlines):
				line = f.readline()
				if len(line) < 4:
					break
				lineObj = json.loads(line)
				rawInputDataObj = {'u':lineObj['reviewerID'], 'asin':lineObj['asin']}
				rawOutputDataObj = clean(lineObj['reviewText'])
				rawInputData.append(rawInputDataObj)
				rawOutputData.append(rawOutputDataObj)
		if save:
			pickle.dump((rawInputData, rawOutputData), open(fileName,'wb'))
	return rawInputData, rawOutputData

def initVecData(rawInputData, rawOutputData, fileName='vecData.p', save=True):
	try:
		vecInputData, vecOutputData = pickle.load(open(fileName,'rb'))
		print('loaded saved vectorized data')
	except (OSError, IOError) as e:
		print('initializing vectorized data')
		dictVect = DictVectorizer()
		vecInputData = dictVect.fit_transform(rawInputData).toarray()
		vecOutputData = [matrix_2_avg(seq_2_matrix(review, embedding_map)) for review in rawOutputData]
		if save:
			pickle.dump((vecInputData, vecOutputData), open(fileName,'wb'))
	return vecInputData, vecOutputData

embedding_map = initEmbeddingMap()

rawInputData, rawOutputData = initRawData(input_file='sanity.json', maxlines=50, save=False)
vecInputData, vecOutputData = initVecData(rawInputData, rawOutputData, save=False)

input_dimensionality = vecInputData.shape[1]
hidden_size = 4
embedding_dimensionality = len(embedding_map.get("the"))

inputs = np.array(vecInputData)
outputs = np.array(vecOutputData)
trainingN = 13
# display_wordvecs(outputs)
# display_wordvecs_similarity(outputs)

model = Sequential()
model.add(Dense(hidden_size, activation='tanh', input_shape=(input_dimensionality,)))
# model.add(Dense(100, activation='relu'))
model.add(Dense(embedding_dimensionality))
model.compile(optimizer='adam', loss='mse')
print(inputs)
print(outputs)

baseline = np.mean(outputs[:trainingN], 0)
baseline_train_mse = np.mean((baseline - outputs[:trainingN]) ** 2)
print("baseline train mse: " + str(baseline_train_mse))
baseline_test_mse = np.mean((baseline - outputs[trainingN:]) ** 2)
print("baseline test mse: " + str(baseline_test_mse))

model.fit(inputs[:trainingN], outputs[:trainingN], epochs=3500, batch_size=32, verbose=0)
train_score = model.evaluate(inputs[:trainingN], outputs[:trainingN], batch_size=32, verbose=0)
print("model train mse: " + str(train_score))
test_score = model.evaluate(inputs[trainingN:], outputs[trainingN:], batch_size=32, verbose=0)
print("model test mse: " + str(test_score))


print(baseline)

