from scipy import spatial
import matplotlib.pyplot as plt
from keras import backend as K

def get_layer_output(model, input_layers_i, output_layer_i, input_data):
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