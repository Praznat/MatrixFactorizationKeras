import sys
import json
import numpy as np
import pickle
from collections import defaultdict
from densesubgraphfinder import DenseSubgraphFinder
from sklearn.feature_extraction import DictVectorizer
import csv
import os

def e(v, d=2):
	return np.matrix.round(v, d)
	
def clean(text):
	from keras.preprocessing.text import text_to_word_sequence
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
	return np.mean(emb_matrix, 0) if len(emb_matrix) else None

def initEmbeddingMap(fileName='embeddingDict.p'):
	try:
		print('loading saved embedding dictionary')
		embedding_map = pickle.load(open(fileName,'rb'))
	except (OSError, IOError) as e:
		print('constructing embeloaddding dictionary')
		embedding_map = {}
		with open('glove.6B.50d.txt') as glove:
			for line in glove:
			    values = line.split()
			    word = values[0]
			    value = np.asarray(values[1:], dtype='float32')
			    embedding_map[word] = value
		pickle.dump(embedding_map, open(fileName,'wb'))
	return embedding_map

class SparcityTarget():
	def __init__(self, n_reviews=0, density_target=0):
		self.n_reviews = n_reviews
		self.density_target = density_target
		self.dsf = DenseSubgraphFinder()
	def update(self, user, item):
		self.dsf.addEdge(user, item)
	def filter(self, rawInputData, rawOutputData):
		if self.n_reviews:
			self.dsf.purge(self.n_reviews)
		if self.density_target:
			self.dsf.condense(self.density_target)
		filteredInput = []
		filteredOutput = []
		for i in range(len(rawInputData)):
			in_datum = rawInputData[i]
			if in_datum['u'] in self.dsf.nodes and in_datum['asin'] in self.dsf.nodes:
				filteredInput.append(in_datum)
				filteredOutput.append(rawOutputData[i])
		return filteredInput, filteredOutput
	def __str__(self):
		return str(self.dsf)

def initRawData(input_file, maxlines = sys.maxsize, sparcity_target = None, fileName='rawData.p', save=True):
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
				user = lineObj['reviewerID']
				item = lineObj['asin']
				rawInputDataObj = {'u':user, 'asin':item}
				rawOutputDataObj = clean(lineObj['reviewText'])
				rawInputData.append(rawInputDataObj)
				rawOutputData.append(rawOutputDataObj)
				if sparcity_target is not None:
					sparcity_target.update(user, item)
		if sparcity_target is not None:
			rawInputData, rawOutputData = sparcity_target.filter(rawInputData, rawOutputData)
		if save:
			pickle.dump((rawInputData, rawOutputData), open(fileName,'wb'))
	return rawInputData, rawOutputData

def getSetFromData(key, data):
	result = set()
	for datum in data:
		result.add(datum.get(key))
	return result

def getSparcityInfo(inputData):
	users = {}
	items = {}
	for datum in inputData:
		u = datum['u']
		i = datum['asin']
		users.setdefault(u, []).append(i)
		items.setdefault(i, []).append(u)
	return (users, items)

def getEmbeddingDim(embeddingData):
	if len(embeddingData) == 0:
		return 0
	i = 0
	while embeddingData[i] is None:
		i += 1
	return embeddingData[i].shape[0]

def initVecData(rawInputData, rawOutputData=[], embedding_map={}, fileName='vecData.p', save=True):
	try:
		vecInputData, vecOutputData = pickle.load(open(fileName,'rb'))
		print('loaded saved vectorized data')
	except (OSError, IOError) as e:
		print('initializing vectorized data')
		dictVect = DictVectorizer()
		vecInputData = dictVect.fit_transform(rawInputData).toarray()
		vecOutputData = None
		if rawOutputData:
			vecOutputData = [matrix_2_avg(seq_2_matrix(review, embedding_map)) for review in rawOutputData]
			dim = getEmbeddingDim(vecOutputData)
			# zero vector for no embedding
			vecOutputData = [v if v is not None and v.shape[0] == dim else None for v in vecOutputData]
			# vecOutputData = [v if v is not None and v.shape[0] == dim else np.zeros(dim) for v in vecOutputData]
		if save:
			pickle.dump((vecInputData, vecOutputData), open(fileName,'wb'))
	return vecInputData, vecOutputData

def to_vec_data(ids):
	return DictVectorizer().fit_transform(ids).toarray()

class MatrixInput():
	def __init__(self, key):
		self.key = key
		self.trainN = 0
		self.totalN = 0
		self.rep = []
	def add(self, matrix, is_train):
		if matrix.shape[0] > 0:
			self.rep.append(matrix)
			self.totalN += len(matrix)
			if is_train:
				self.trainN += len(matrix)
	def get_np_vec(self, only_training):
		return np.vstack(self.rep[:self.trainN]) if only_training else np.vstack(self.rep)
	def get_seq_len(self, only_training):
		return self.trainN if only_training else self.totalN
	def __str__(self):
		return self.key + " " + str(self.get_np_vec(False).shape) + " trn:" + str(self.trainN)

def initMatInputData(rawInputData, rawOutputData, embedding_map, trainingP, fileName='matData.p', save=True, extra_info={}):
	try:
		matUserInputData, matItemInputData = pickle.load(open(fileName,'rb'))
		print('loaded saved matrix data')
	except (OSError, IOError) as e:
		print('initializing matrix data')
		if len(rawInputData) != len(rawOutputData):
			raise ValueError("Need same size of input and output")
		trainingN = int(len(rawInputData) * trainingP) if type(trainingP) is float else trainingP
		users = {}
		items = {}
		dictVect = DictVectorizer()
		for i in range(len(rawInputData)):
			vecOutput = seq_2_matrix(rawOutputData[i], embedding_map)
			rawInput = rawInputData[i]
			user = rawInput['u']
			item = rawInput['asin']
			users.setdefault(user, MatrixInput(user)).add(vecOutput, i < trainingN)
			items.setdefault(item, MatrixInput(item)).add(vecOutput, i < trainingN)
		matUserInputData = []
		matItemInputData = []
		extra_info['user_seq_sizes'] = [m.get_seq_len(False) for m in users.values()]
		extra_info['item_seq_sizes'] = [m.get_seq_len(False) for m in items.values()]
		print(extra_info)
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

def debug_get_rating(u, asin, filename, maxlines = sys.maxsize):
	with open(filename,'r') as f:
		for i in range(maxlines):
			line = f.readline()
			if len(line) < 4:
				break
			terms = line.split(',')
			if u == terms[0] and asin == terms[1]:
				return terms[2]

def initRatingsOutputData(rawInputData, input_file, maxlines = sys.maxsize, fileName='ratingsData.p', save=True):
	try:
		ratingsData = pickle.load(open(fileName,'rb'))
		print('loaded saved ratings data')
	except (OSError, IOError) as e:
		hasReviewData = []
		userItemDict = {}
		noReviewData = []
		# just fills in ratings for known reviews. doesnt return ratings with no review
		for i in range(len(rawInputData)):
			rawInput = rawInputData[i]
			userItem = toKey(rawInput['u'], rawInput['asin'])
			userItemDict[userItem] = i
			hasReviewData.append(None)
		with open(input_file,'r') as f:
			for i in range(maxlines):
				line = f.readline()
				if len(line) < 4:
					break
				user, item, rating, time = line.split(',')
				rating = float(rating)
				k = userItemDict.get(toKey(user, item))
				if k is not None:
					hasReviewData[k] = rating
			failure = None in hasReviewData
			if failure:
				raise ValueError(str(len([r for r in hasReviewData if r is None])) + " reviews did not have corresponding rating.")
		ratingsData = hasReviewData + noReviewData
		if save:
			pickle.dump(ratingsData, open(fileName,'wb'))
	return ratingsData

def initRatingsInputOutputData(input_file, maxlines = sys.maxsize, fileName='ratingsIOData.p', save=True):
	try:
		ratingsInputData, ratingsOutputData = pickle.load(open(fileName,'rb'))
		print('loaded saved ratings IO data')
	except (OSError, IOError) as e:
		ratingsInputData = []
		ratingsOutputData = []
		with open(input_file,'r') as f:
			for i in range(maxlines):
				line = f.readline()
				if len(line) < 4:
					break
				user, item, rating, time = line.split(',')
				rating = float(rating)
				ratingsInputData.append({'u':user, 'asin':item})
				ratingsOutputData.append(rating)
		if save:
			pickle.dump((ratingsInputData, ratingsOutputData), open(fileName,'wb'))
	return ratingsInputData, ratingsOutputData

def corrScore(v1, v2, nanval=0):
	i = np.where(v1 * v2)[0]
	c = np.corrcoef(v1[i], v2[i])[0,1]
	return c if not np.isnan(c) else nanval

def mseScore(v1, v2):
	i = np.where(v1 * v2)[0]
	return np.mean((v1[i] - v2[i])**2)

def perUserPrediction(vecUsers, ratings):
	sumRPerUser = np.sum(np.transpose(vecUsers) * ratings, axis=1)
	nRPerUser = np.sum(vecUsers,axis=0)
	avgRPerUser = sumRPerUser / nRPerUser
	return avgRPerUser
	# return np.sum(vecUsers * avgRPerUser, axis=1)

def perUserScore(vecUsers, predR, trueR, scoreFn=mseScore):
	vu = np.transpose(vecUsers)
	score = [scoreFn(v * trueR, v * predR) for v in vu if np.sum(v) != 0.0]
	return np.mean(score)

def toCsv(data, filename):
	with open(filename + '.csv', 'w', newline='') as csvf:
		csvwriter = csv.writer(csvf, delimiter=',')
		for row in data:
			csvwriter.writerow(row)
        


