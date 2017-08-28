import models
import models_mcmc
import utils
import numpy as np
import keras
import diagnostics
from utils import e
import matplotlib.pyplot as plt
from keras import metrics
from keras import backend as K
from keras.callbacks import EarlyStopping

def evaluate(model, train_inputs, train_outputs, test_inputs, test_outputs):
	model_train_mse = model.evaluate(train_inputs, train_outputs, verbose=0)
	print(test_outputs)
	baseline = np.mean(train_outputs)
	baseline_train_mse = np.mean((baseline - train_outputs) ** 2)
	print("baseline train mse: " + str(baseline_train_mse))
	print("model train mse: " + str(model_train_mse))

	model_test_mse = model.evaluate(test_inputs, test_outputs, verbose=1, batch_size=len(test_outputs))
	baseline_test_mse = np.mean((baseline - test_outputs) ** 2)
	print("baseline test mse: " + str(baseline_test_mse))
	print("model test mse: " + str(model_test_mse))

	jitter = lambda x : np.random.normal(x, np.std(x)/10)

	predictions = model.predict(test_inputs, verbose=0)
	tf_session = K.get_session()
	print(np.mean(metrics.mean_squared_error(predictions, test_outputs).eval(session=tf_session)))
	print(np.mean((predictions - test_outputs) ** 2))
	plt.scatter(jitter(predictions), jitter(test_outputs))
	plt.show()

def user_item_to_embedding(vecInputData, vecOutputData, hidden_size=4, epochs=3500):
	input_dimensionality = vecInputData.shape[1]
	embedding_dimensionality = vecOutputData[0].shape[0]

	inputs = np.asarray(vecInputData)
	outputs = np.asarray(vecOutputData)
	trainingN = 13
	# diagnostics.display_wordvecs(outputs)
	# diagnostics.display_wordvecs_similarity(outputs)

	model = models.create_dumb_model(input_dimensionality, hidden_size, embedding_dimensionality)

	baseline = np.mean(outputs[:trainingN], 0)
	baseline_train_mse = np.mean((baseline - outputs[:trainingN]) ** 2)
	baseline_test_mse = np.mean((baseline - outputs[trainingN:]) ** 2)

	model.fit(inputs[:trainingN], outputs[:trainingN], epochs=epochs, batch_size=32, verbose=0)
	train_score = model.evaluate(inputs[:trainingN], outputs[:trainingN], batch_size=32, verbose=0)
	test_score = model.evaluate(inputs[trainingN:], outputs[trainingN:], batch_size=32, verbose=0)

	print("baseline train mse: " + str(baseline_train_mse))
	print("baseline test mse: " + str(baseline_test_mse))
	print("model train mse: " + str(train_score))
	print("model test mse: " + str(test_score))

def natlang_dawidskene(vecUsers, vecItems, vecOutputData, hidden_size=1, epochs=3500):
	embedding_dimensionality = vecOutputData[0].shape[0]
	outputs = np.asarray(vecOutputData)

	base_model = models.create_u_agnostic_model_sane(vecItems.shape[1], embedding_dimensionality)
	base_model.summary()
	base_model.fit([vecItems, outputs], outputs, epochs=epochs)
	init_weights = base_model.layers[1].get_weights()
	predicted_embedding_avg = diagnostics.get_layer_output(base_model, [0], 1, [vecItems])

	model = models.create_nlds_model(vecUsers.shape[1], vecItems.shape[1], hidden_size, embedding_dimensionality)
	model.layers[3].set_weights(init_weights)

	init_iError = diagnostics.get_layer_output(model, [1, 4], 6, [vecItems, outputs])
	print(np.mean(init_iError, axis=1))

	model.fit([vecUsers, vecItems, outputs], np.zeros((len(outputs), embedding_dimensionality)), epochs=epochs)
	model.summary()
	print(model.layers[2].get_weights())
	uNoise = diagnostics.get_layer_output(model, [0], 2, [vecUsers])
	print(uNoise)
	predicted_embedding = diagnostics.get_layer_output(model, [1], 3, [vecItems])
	iError = diagnostics.get_layer_output(model, [1, 4], 6, [vecItems, outputs])
	print(np.mean(iError, axis=1))
	diagnostics.display_wordvecs_sqeuclidean(outputs, predicted_embedding_avg)
	diagnostics.display_wordvecs_sqeuclidean(outputs, predicted_embedding)

def natlang_dawidskene_mc(vecUsers, vecItems, vecOutputData, hidden_size=1, epochs=2000, tune=1000):
	embedding_dimensionality = vecOutputData[0].shape[0]
	outputs = np.asarray(vecOutputData)
	model = models_mcmc.NLDSModelMC(vecUsers.shape[1], vecItems.shape[1], hidden_size, embedding_dimensionality)
	model.fit(vecUsers, vecItems, outputs, epochs, tune)

def natlang_u_agnostic(vecItems, vecOutputData, epochs=2000, sanity=True):
	embedding_dimensionality = vecOutputData[0].shape[0]
	outputs = np.asarray(vecOutputData)
	model = None
	if sanity:
		model = models.create_u_agnostic_model_sane(vecItems.shape[1], embedding_dimensionality)
		model.fit([vecItems, outputs], outputs, epochs=epochs)
	else:
		model = models.create_u_agnostic_model(vecItems.shape[1], embedding_dimensionality)
		model.fit([vecItems, outputs], np.zeros((len(outputs), embedding_dimensionality)), epochs=epochs)
	print(model.summary())
	predicted_embedding = diagnostics.get_layer_output(model, [0,2], 1, [vecItems])
	iError = diagnostics.get_layer_output(model, [0,2], 3, [vecItems, outputs])
	print(e(predicted_embedding))
	print("***")
	print(e(outputs))
	print(np.mean(iError, axis=1)) # TODO these norms arent good indicator of error
	diagnostics.display_wordvecs_similarity(outputs, predicted_embedding)

def matrix_factorization(vecUsers, vecItems, ratingsData, hidden_size=4, epochs=3500, training=0.8):
	model = models.create_factorization_model(vecUsers.shape[1], vecItems.shape[1], hidden_size)

	trainingN = int(len(ratingsData) * training) if type(training) is float else training

	train_inputs = [vecUsers[:trainingN], vecItems[:trainingN]]
	train_outputs = ratingsData[:trainingN]
	test_inputs = [vecUsers[trainingN:], vecItems[trainingN:]]
	test_outputs = ratingsData[trainingN:]
	print(model.summary())

	early_stopping = EarlyStopping(monitor='loss', patience=8)
	early_stopping_val = EarlyStopping(monitor='val_loss', patience=12)
	batch_size = 32
	model.fit(train_inputs, train_outputs, validation_split=0.2, callbacks=[early_stopping, early_stopping_val], batch_size=batch_size, epochs=epochs)
	evaluate(model, train_inputs, train_outputs, test_inputs, test_outputs)


def deepconn_1st_order(matUserInputData, matItemInputData, ratingsData, u_seq_len=200, i_seq_len=200, hidden_size=4, epochs=3500, training=None):
	embedding_dimensionality = matUserInputData[0].shape[1]
	deepconn = models.DeepCoNN(embedding_dimensionality, hidden_size, u_seq_len, i_seq_len)
	derpcon = models.DerpCon(embedding_dimensionality, hidden_size, u_seq_len, i_seq_len)

	model = deepconn.create_deepconn_dp()
	# model = derpcon.create_derpcon_dp()

	user_input = keras.preprocessing.sequence.pad_sequences(np.asarray(matUserInputData), maxlen=u_seq_len)
	item_input = keras.preprocessing.sequence.pad_sequences(np.asarray(matItemInputData), maxlen=i_seq_len)
	
	trainingN = int(len(user_input) * training) if type(training) is float else training

	inputs = [user_input, item_input]
	outputs = np.asarray(ratingsData)
	print(model.summary())

	train_inputs = [user_input[:trainingN], item_input[:trainingN]]
	train_outputs = outputs[:trainingN]
	test_inputs = [user_input[trainingN:], item_input[trainingN:]]
	test_outputs = outputs[trainingN:]

	early_stopping = EarlyStopping(monitor='loss', patience=4)
	early_stopping_val = EarlyStopping(monitor='val_loss', patience=6)
	batch_size = 32
	model.fit(train_inputs, train_outputs, validation_split=0.2, callbacks=[early_stopping, early_stopping_val], batch_size=batch_size, epochs=epochs)
	evaluate(model, train_inputs, train_outputs, test_inputs, test_outputs)




