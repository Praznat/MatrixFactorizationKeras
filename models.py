import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Embedding, Lambda, RepeatVector, Merge
from keras.layers.merge import Add, Multiply, Dot, Concatenate
from keras import regularizers
from keras import backend as K

# import numpy as np
# from keras import backend as K
# tmp1 = np.array([[1,0,1,0,0],[1,0,1,0,0]])
# tmp2 = np.array([[1,0,0,1,1],[1,0,0,.5,0]])
# input1 = Input(shape=(5,))
# input2 = Input(shape=(5,))
# lamb = Lambda(lambda x: (x[0]-x[1])**2)([input1, input2])
# model = Model(inputs=[input1, input2], outputs=[lamb])
# model.compile(optimizer='adam', loss='mse')
# print(model.summary())
# get_layer_output_fn = K.function([model.layers[i].input for i in [0,1]], [model.layers[2].output])
# layer_output = get_layer_output_fn([tmp1,tmp2])
# print(layer_output)

# USELESS
def simplex_reg(weight_matrix):
	return 0.01 * (K.abs(K.sum(weight_matrix) - 1.))

def ErrorLayer_BK(v1, v2):
	return keras.layers.merge([v1, v2], mode='cos')

def ErrorLayer(v1, v2):
	return Lambda(lambda x: (x[0]-x[1])**2, name="error")([v1, v2])

def create_dumb_model(input_size, hidden_size, output_embedding_size):
	model = Sequential()
	model.add(Dense(hidden_size, activation='tanh', input_shape=(input_size,)))
	model.add(Dense(output_embedding_size)) # activation=None: linear activation
	model.compile(optimizer='adam', loss='mse')
	return model

def create_u_agnostic_model(item_size, output_embedding_size):
	inputI = Input(shape=(item_size,), name="item_1hot")
	inputE = Input(shape=(output_embedding_size,), name="target_embedding")
	hiddenI = Dense(output_embedding_size, activation=None, name="inferred_embedding")(inputI) # linear activation for embeddings
	error = ErrorLayer(hiddenI, inputE)
	model = Model(inputs=[inputI, inputE], outputs=[error])
	model.compile(optimizer='adam', loss='mse')
	return model

def create_u_agnostic_model_sane(item_size, output_embedding_size):
	inputI = Input(shape=(item_size,), name="item_1hot")
	inputE = Input(shape=(output_embedding_size,), name="target_embedding")
	hiddenI = Dense(output_embedding_size, activation=None, name="inferred_embedding")(inputI) # linear activation for embeddings
	error = ErrorLayer(hiddenI, inputE)
	output = Lambda(lambda x: x[0], name="output")([hiddenI, error])
	model = Model(inputs=[inputI, inputE], outputs=[output])
	model.compile(optimizer='adam', loss='mse')
	return model

def create_nlds_model(user_size, item_size, hidden_size, output_embedding_size):
	inputU = Input(shape=(user_size,), name="user_1hot")
	inputI = Input(shape=(item_size,), name="item_1hot")
	inputE = Input(shape=(output_embedding_size,), name="target_embedding")
	hiddenI = Dense(output_embedding_size, activation=None, name="inferred_embedding")(inputI) # linear activation for embeddings
	errorIE = ErrorLayer(hiddenI, inputE)
	HIDDEN_SIZE_1 = 1
	hiddenU = Dense(HIDDEN_SIZE_1, activation='sigmoid', use_bias=False, activity_regularizer=regularizers.l1(.001))(inputU) # user vars represent added noise! 0 is good. like An's work (use tanh?)
	# regularizer weight of 0.001 seems to work well... (typical errorIE is about 0.02-0.04)
	errorU = RepeatVector(output_embedding_size)(hiddenU)
	errorU = Flatten(name = 'user_noise')(errorU)

	output = Lambda(lambda x: x[0] * (1 - x[1]), name="output")([errorIE, errorU])

	model = Model(inputs=[inputU, inputI, inputE], outputs=[output])
	model.compile(optimizer='adam', loss='mse')
	return model

def create_nlds_model_BK(user_size, item_size, hidden_size, output_embedding_size, extra=True):
	inputU = Input(shape=(user_size,))
	inputI = Input(shape=(item_size,))
	hiddenU = Dense(hidden_size, activation='sigmoid', use_bias=False, activity_regularizer=regularizers.l1(.01))(inputU) # user vars represent added noise! 0 is good. like An's work (use tanh?)
	if extra:
		hiddenU = Concatenate()([hiddenU, inputI])
	hiddenI = Dense(output_embedding_size, activation=None)(inputI) # linear activation for embeddings
	noise = Dense(output_embedding_size, activation=None, use_bias=False, activity_regularizer=regularizers.l1(.01))(hiddenU) # noise added to 'true' embedding to get observed one
	output = Add()([hiddenI, noise])
	model = Model(inputs=[inputU, inputI], outputs=[output])
	model.compile(optimizer='adam', loss='mse')
	return model

def create_aspect_model(user_size, item_size, hidden_size, output_embedding_size):
	inputU = Input(shape=(user_size,))
	inputI = Input(shape=(item_size,))
	hiddenU = Dense(hidden_size, activation='tanh')(inputU)
	hiddenI = Dense(hidden_size, activation='tanh')(inputI)
	aspectVals = Multiply()([hiddenU, hiddenI])
	output = Dense(output_embedding_size)(aspectVals)
	model = Model(inputs=[inputU, inputI], outputs=[output])
	model.compile(optimizer='adam', loss='mse')
	return model

class DeepCoNN():
	def __init__(self, embedding_size, hidden_size, max_seq_len, filters=100, kernel_size=3):
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.max_seq_len = max_seq_len
		self.filters = filters
		self.kernel_size = kernel_size
		self.inputU, self.towerU = self.create_deepconn_tower()
		self.inputI, self.towerI = self.create_deepconn_tower()
		self.joined = Concatenate()([self.towerU, self.towerI])
		self.outNeuron = output = Dense(1)(self.joined)

	def create_deepconn_tower(self):
		input_layer = Input(shape=(self.max_seq_len, self.embedding_size))
		tower = Conv1D(filters=self.filters, kernel_size=self.kernel_size)(input_layer)
		tower = MaxPooling1D()(tower)
		tower = Flatten()(tower)
		tower = Dense(self.hidden_size)(tower) # wait so linear activations??
		return input_layer, tower

	def create_1st_order_only(self):
		model = Model(inputs=[self.inputU, self.inputI], outputs=[self.outNeuron])
		model.compile(optimizer='adam', loss='mse')
		return model

	def create_deepconn_dp(self):
		dotproduct = Dot(axes=1)([self.towerU, self.towerI])
		output = Add()([self.outNeuron, dotproduct])
		model = Model(inputs=[self.inputU, self.inputI], outputs=[output])
		model.compile(optimizer='adam', loss='mse')
		return model