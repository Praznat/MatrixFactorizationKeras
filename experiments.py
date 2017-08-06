import models
import utils
import numpy as np
import keras
import diagnostics
from utils import e

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

def deepconn_1st_order(matUserInputData, matItemInputData, ratingsData, max_seq_len=200, hidden_size=4, epochs=3500):
	embedding_dimensionality = matUserInputData[0].shape[1]
	deepconn = models.DeepCoNN(embedding_dimensionality, hidden_size, max_seq_len, filters=1)

	# model = deepconn.create_1st_order_only()
	model = deepconn.create_deepconn_dp()

	user_input = keras.preprocessing.sequence.pad_sequences(np.asarray(matUserInputData), maxlen=max_seq_len)
	item_input = keras.preprocessing.sequence.pad_sequences(np.asarray(matItemInputData), maxlen=max_seq_len)
	
	trainingN = 13

	inputs = [user_input, item_input]
	outputs = np.asarray(ratingsData)
	print(model.summary())

	model.fit(inputs, outputs, epochs=epochs)
	baseline = np.mean(outputs)
	baseline_train_mse = np.mean((baseline - outputs) ** 2)
	print("baseline train mse: " + str(baseline_train_mse))


