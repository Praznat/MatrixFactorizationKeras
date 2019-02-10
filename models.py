import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, AveragePooling1D, Flatten, Reshape, Embedding, Lambda, RepeatVector, Merge
from keras.layers.merge import Add, Multiply, Dot, Concatenate
from keras import regularizers
from keras import backend as K

def to_one(m):
	return lambda x: (m * (K.sum(K.abs(x - 1.))))

def create_user_item_model(user_size, item_size):
	''' same as factorization model with hidden_size=0 '''
	inputU = Input(shape=(user_size,))
	inputI = Input(shape=(item_size,))
	output = Dense(1, use_bias=True)(Concatenate()([inputU, inputI]))
	model = Model(inputs=[inputU, inputI], outputs=[output])
	model.compile(optimizer='adam', loss='mse')
	return model

def create_factorization_model(user_size, item_size, hidden_size, **kwargs):
	''' Basic generalization of matrix factorization models.
	user_size and item_size are number of users and items, respectively
	hidden_size determines number of attributes
	optional arguments:
	regularization: amount of L2 regularization to apply, none by default
	activations: vector of activations for user and item hidden layers. default is "linear", use "relu" for non-negative matrix factorization
	more_complex: learns additional weights for each attribute instead of taking simple dot product. False by default.
	useIntercepts: whether to use user and item intercepts. common practice, but false by default here
	squash: if output is bounded, it can help to tell that to the model, but then need to normalize output to fall between 0 and 1. False by default.
	'''
	inputU = Input(shape=(user_size,), name="user_1hot")
	inputI = Input(shape=(item_size,), name="item_1hot")
	regularization = kwargs.get('regularization')
	regularizer = regularizers.l2(regularization) if regularization else None
	if hidden_size:
		activations = kwargs.get('activations') or ["linear","linear"]
		print(activations)
		hiddenU = Dense(hidden_size, activation=activations[0], name="user_hidden", kernel_regularizer=regularizer, use_bias=True)(inputU)
		hiddenI = Dense(hidden_size, activation=activations[1], name="item_hidden", kernel_regularizer=regularizer, use_bias=True)(inputI)
		output = Dense(1, kernel_regularizer=to_one(regularization or .01))(Multiply(name="aspect_points")([hiddenU, hiddenI])) if kwargs.get('more_complex') else Dot(axes=1)([hiddenU, hiddenI])
		if kwargs.get('useIntercepts'):
			intercept = Dense(1, use_bias=True, kernel_regularizer=regularizer)(Concatenate()([inputU, inputI]))
			output = Add(name="prediction")([output, intercept])
	else:
		output = Dense(1, name="prediction", use_bias=True, kernel_regularizer=regularizer)(Concatenate()([inputU, inputI])) # same as user_item model
	if kwargs.get('squash'):
		output = Dense(1)(Dense(1, activation="sigmoid")(output))
	model = Model(inputs=[inputU, inputI], outputs=[output])
	model.compile(optimizer='adam', loss='mse')
	return model

def Sum(inLayer, inLayerSize, name="Sum"):
	return Flatten()(AveragePooling1D(pool_size=inLayerSize, name=name)(Reshape((inLayerSize,1))(inLayer)))

def create_joint_model(user_size, item_size, hidden_size, output_embedding_size, **kwargs):
	'''
	Matrix factorization model regularized by text embedding likelihood (from reviews).
	user_size and item_size are number of users and items, respectively
	output_embedding_size is dimensionality of the word vectors (embeddings)
	hidden_size determines number of attributes
	optional arguments:
	activations: vector of activations for user and item hidden layers. default is "linear", use "relu" for non-negative matrix factorization
	more_complex: learns additional weights for each attribute instead of taking simple dot product. False by default.
	useIntercepts: whether to use user and item intercepts. common practice, but false by default here
	squash: if ratings are bounded, it can help to tell that to the model, but then need to normalize ratings to fall between 0 and 1. False by default.
	'''
	inputU = Input(shape=(user_size,), name="user_1hot")
	inputI = Input(shape=(item_size,), name="item_1hot")
	activations = kwargs.get('activations') or ["linear","linear"]
	hiddenU = Dense(hidden_size, activation=activations[0], name="user_hidden")(inputU)
	hiddenI = Dense(hidden_size, activation=activations[1], name="item_hidden")(inputI)
	aspectVals = Multiply(name="aspect_points")([hiddenU, hiddenI])
	h = hidden_size
	if kwargs.get('useIntercepts'):
		intercept = Dense(1, use_bias=False, name="intercept")(Concatenate()([inputU, inputI]))
		aspectVals = Concatenate(name="features")([aspectVals, intercept])
		h += 1
	embedding = Dense(output_embedding_size, name="embedding")(aspectVals)
	rating = Dense(1)(aspectVals, name="rating") if kwargs.get('more_complex') else Sum(aspectVals, h, "rating")
	if kwargs.get('squash'):
		rating = Dense(1)(Dense(1, activation="sigmoid")(rating))
	model = Model(inputs=[inputU, inputI], outputs=[embedding, rating])
	model_textless = Model(inputs=[inputU, inputI], outputs=[rating])
	model.compile(optimizer='adam', loss='mse')
	model_textless.compile(optimizer='adam', loss='mse')
	return model, model_textless

class DeepCoNN():
	''' DeepCoNN text-based factorization model as described in the paper '''
	# def __init__(self, embedding_size, hidden_size, u_seq_len, i_seq_len, filters=100, kernel_size=3, strides=1): # original DeepCoNN paper settings
	def __init__(self, embedding_size, hidden_size, u_seq_len, i_seq_len, filters=2, kernel_size=3, strides=6):
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.filters = filters
		self.kernel_size = kernel_size
		self.inputU, self.towerU = self.create_deepconn_tower(u_seq_len)
		self.inputI, self.towerI = self.create_deepconn_tower(i_seq_len)
		self.joined = Concatenate()([self.towerU, self.towerI])
		self.outNeuron = Dense(1)(self.joined)

	def create_deepconn_tower(self, max_seq_len):
		input_layer = Input(shape=(max_seq_len, self.embedding_size))
		tower = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation="relu")(input_layer)
		tower = MaxPooling1D()(tower)
		tower = Flatten()(tower)
		tower = Dense(self.hidden_size, activation="relu")(tower)
		return input_layer, tower

	def create_1st_order_only(self):
		model = Model(inputs=[self.inputU, self.inputI], outputs=[self.outNeuron])
		model.compile(optimizer='adam', loss='mse')
		return model

	def create_deepconn_dp(self):
		''' simple dot product instead of factorization machine for final layer.
		this simplification yielded similar results in the paper and should work
		better on small data due to less overfitting. '''
		dotproduct = Dot(axes=1)([self.towerU, self.towerI])
		output = Add()([self.outNeuron, dotproduct])
		model = Model(inputs=[self.inputU, self.inputI], outputs=[output])
		model.compile(optimizer='adam', loss='mse')
		return model

class DerpCon():
	''' similar to DeepCoNN, except simpler: uses average embeddings in place of learning a CNN '''
	def __init__(self, embedding_size, hidden_size, u_seq_len, i_seq_len):
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.inputU, self.towerU = self.create_derpcon_tower(u_seq_len)
		self.inputI, self.towerI = self.create_derpcon_tower(i_seq_len)
		self.joined = Concatenate()([self.towerU, self.towerI])
		self.outNeuron = output = Dense(1)(self.joined)
	def create_derpcon_tower(self, max_seq_len):
		input_layer = Input(shape=(max_seq_len, self.embedding_size))
		tower = AveragePooling1D(pool_size=max_seq_len)(input_layer)
		tower = Flatten()(tower)
		tower = Dense(self.hidden_size, activation="relu")(tower)
		return input_layer, tower
	def create_derpcon_dp(self):
		dotproduct = Dot(axes=1)([self.towerU, self.towerI])
		output = Add()([self.outNeuron, dotproduct])
		model = Model(inputs=[self.inputU, self.inputI], outputs=[output])
		model.compile(optimizer='adam', loss='mse')
		return model
