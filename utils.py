import sys
import json
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from scipy import spatial

from keras.preprocessing.text import text_to_word_sequence

def e(v, d=2):
	return np.matrix.round(v, d)
	
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

def getSetFromData(key, data):
	result = set()
	for datum in data:
		result.add(datum.get(key))
	return result

def initVecData(rawInputData, rawOutputData, embedding_map, fileName='vecData.p', save=True):
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

def initMatInputData(rawInputData, rawOutputData, embedding_map, fileName='matData.p', save=True):
	try:
		matUserInputData, matItemInputData = pickle.load(open(fileName,'rb'))
		print('loaded saved matrix data')
	except (OSError, IOError) as e:
		print('initializing matrix data')
		if len(rawInputData) != len(rawOutputData):
			raise ValueError("Need same size of input and output")
		users = {}
		items = {}
		dictVect = DictVectorizer()
		for i in range(len(rawInputData)):
			vecOutput = seq_2_matrix(rawOutputData[i], embedding_map)
			rawInput = rawInputData[i]
			user = rawInput['u']
			item = rawInput['asin']
			users.setdefault(user, []).append(vecOutput)
			items.setdefault(item, []).append(vecOutput)
		matUserInputData = []
		matItemInputData = []
		users = {k: np.vstack(v) for k, v in users.items()}
		items = {k: np.vstack(v) for k, v in items.items()}
		for i in range(len(rawInputData)):
			rawInput = rawInputData[i]
			user = rawInput['u']
			item = rawInput['asin']
			matUserInputData.append(users.get(user))
			matItemInputData.append(items.get(item))
		if save:
			pickle.dump((matUserInputData, matItemInputData), open(fileName,'wb'))
	return matUserInputData, matItemInputData

def toKey(user, item):
	return (user, item)

def initRatingsOutputData(rawInputData, input_file, maxlines = sys.maxsize, fileName='ratingsData.p', save=True):
	try:
		ratingsData = pickle.load(open(fileName,'rb'))
		print('loaded saved ratings data')
	except (OSError, IOError) as e:
		ratingsData = []
		userItemDict = {}
		for i in range(len(rawInputData)):
			rawInput = rawInputData[i]
			userItem = toKey(rawInput['u'], rawInput['asin'])
			userItemDict[userItem] = i
			ratingsData.append(None) # check later to make sure no Nones left
		with open(input_file,'r') as f:
			for i in range(maxlines):
				line = f.readline()
				if len(line) < 4:
					break
				terms = line.split(',')
				user = terms[0]
				item = terms[1]
				rating = float(terms[2])
				i = userItemDict.get(toKey(user, item))
				if i is not None:
					ratingsData[i] = rating
			failure = None in ratingsData
			if failure:
				raise ValueError("Some reviews did not have corresponding rating.")
		if save:
			pickle.dump(ratingsData, open(fileName,'wb'))
	return ratingsData



