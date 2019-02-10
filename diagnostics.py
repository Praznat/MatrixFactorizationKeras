from scipy import spatial
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
from collections import defaultdict

def get_layer_output(model, input_layers_names, output_layer_name, input_data):
	inputs = [model.get_layer(name).input for name in input_layers_names]
	get_layer_output_fn = K.function(inputs, [model.get_layer(output_layer_name).output])
	layer_output = get_layer_output_fn(input_data)[0]
	return layer_output

def get_layer_output_i(model, input_layers_i, output_layer_i, input_data):
	inputs = [model.layers[i].input for i in input_layers_i]
	get_layer_output_fn = K.function(inputs, [model.layers[output_layer_i].output])
	layer_output = get_layer_output_fn(input_data)[0]
	return layer_output

def display_wordvecs(wordvecs):
	maxv = np.max(wordvecs)
	plt.imshow(np.array(wordvecs) / maxv, cmap='RdBu', interpolation='nearest')
	plt.show()

def display_intra_wordvecs_similarity(wordvecs):
	display_wordvecs_similarity(wordvecs, wordvecs)

def display_wordvecs_similarity(v1, v2):
	d = [[spatial.distance.cosine(a,b) for a in v1] for b in v2]
	plt.imshow(d, cmap='hot', interpolation='nearest')
	plt.show()
def display_wordvecs_sqeuclidean(v1, v2):
	d = [[spatial.distance.sqeuclidean(a,b) for a in v1] for b in v2]
	plt.imshow(d, cmap='hot', interpolation='nearest')
	plt.show()

def display_useritem_matrix(vusers, vitems):
	n_users = vusers.shape[1]
	n_items = vitems.shape[1]
	print(str(n_users) + " users")
	print(str(n_items) + " items")
	iusers = np.where(vusers)[1]
	iitems = np.where(vitems)[1]
	mat = np.zeros((n_users, n_items))
	for i in range(len(iusers)):
		mat[iusers[i], iitems[i]] = 1
	plt.imshow(mat, cmap='hot', interpolation='nearest')
	plt.show()

def word_analysis(scores, sentences, pctile, prior_denom=5):
	sentences = np.asarray(sentences)
	pctile = pctile/100. if pctile > 1 else pctile
	lo_pctile = np.percentile(scores, 100 * min(pctile, 1 - pctile))
	hi_pctile = np.percentile(scores, 100 - lo_pctile)
	lo_k = np.where(scores <= lo_pctile)[0]
	hi_k = np.where(scores >= hi_pctile)[0]
	lo_sentences = sentences[lo_k]
	hi_sentences = sentences[hi_k]
	lo_words = defaultdict(int)
	hi_words = defaultdict(int)
	word_scores = {}
	word_set = set()
	for sentence in lo_sentences:
		for word in sentence:
			lo_words[word] += 1
			word_set.add(word)
	for sentence in hi_sentences:
		for word in sentence:
			hi_words[word] += 1
			word_set.add(word)
	for word in word_set:
		lo_n = lo_words[word]
		hi_n = hi_words[word]
		word_scores[word] = (hi_n - lo_n) / (hi_n + lo_n + prior_denom)
	return word_scores

