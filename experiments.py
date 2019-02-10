import models
import utils
import numpy as np
import keras
import diagnostics
from utils import e
import matplotlib.pyplot as plt
from keras import metrics
from keras import backend as K
from keras.callbacks import EarlyStopping

jitter = lambda x : np.random.normal(x, np.std(x)/10)
mse = lambda x, y: np.mean((x - y) ** 2)
rmse = lambda x, y: np.sqrt(mse(x, y))

BATCH_SIZE = 64

def evaluate(model, train_inputs, train_outputs, test_inputs, test_outputs, output_i=None, csvname=None,
	metrics=["baseline_train_mse", "model_train_mse", "baseline_test_mse", "model_test_mse",
	"model_test_peruser_mse", "model_test_corr"]):
	model_train_mse = model.evaluate(train_inputs, train_outputs, verbose=0)
	batchsize = len(test_outputs) if output_i is None else len(test_outputs[output_i])
	model_test_mse = model.evaluate(test_inputs, test_outputs, verbose=1, batch_size=batchsize)
	predictions = model.predict(test_inputs, verbose=0)
	if output_i is not None:
		train_outputs = train_outputs[output_i]
		test_outputs = test_outputs[output_i]
		predictions = predictions[output_i]
		model_train_mse = model_train_mse[1 + output_i]
		model_test_mse = model_test_mse[1 + output_i]
	predictions = predictions.reshape(len(predictions))
	train_outputs = np.array(train_outputs).reshape(len(train_outputs))
	test_outputs = np.array(test_outputs).reshape(len(test_outputs))

	results = {}

	if "baseline_train_mse" in metrics:
		baseline = np.mean(train_outputs, axis=0)
		results["baseline_train_mse"] = mse(baseline, train_outputs)

	if "baseline_test_mse" in metrics:
		results["baseline_test_mse"] = mse(baseline, test_outputs)

	if "model_train_mse" in metrics:
		results["model_train_mse"] = model_train_mse

	if "model_test_mse" in metrics:
		results["model_test_mse"] = model_test_mse

	if "model_test_peruser_mse" in metrics:
		results["model_test_peruser_mse"] = utils.perUserScore(test_inputs[0], predictions, test_outputs)

	if "model_test_corr" in metrics:
		results["model_test_corr"] = np.corrcoef(predictions, test_outputs)[0,1]

	if csvname:
		data = np.vstack((predictions, test_outputs))
		utils.toCsv(np.transpose(data), csvname)
	# plt.scatter(predictions, jitter(test_outputs))
	# plt.show()
	return results

def organize_data(vecUsers, vecItems, vecOutputData, ratingsData, trainingP):
	trainingN = int(len(ratingsData) * trainingP) if type(trainingP) is float else trainingP
	embedding_dimensionality = utils.getEmbeddingDim(vecOutputData)

	text_train_i, textless_train_i, text_test_i, textless_test_i = [], [], [], []
	for i in range(trainingN):
		if vecOutputData[i] is not None:
			text_train_i.append(i)
		else:
			textless_train_i.append(i)
	for i in range(trainingN, len(vecOutputData)):
		if vecOutputData[i] is not None:
			text_test_i.append(i)
		else:
			textless_test_i.append(i)

	reshapy = lambda x: np.asarray(list(x))

	D = {}
	D['textless_train_inputs'] = [vecUsers[textless_train_i], vecItems[textless_train_i]]
	D['textless_train_outputs'] = [ratingsData[textless_train_i]]
	D['text_train_inputs'] = [vecUsers[text_train_i], vecItems[text_train_i]]
	D['text_train_outputs'] = [reshapy(vecOutputData[text_train_i]), ratingsData[text_train_i]]
	D['textless_test_inputs'] = [vecUsers[textless_test_i], vecItems[textless_test_i]]
	D['textless_test_outputs'] = [ratingsData[textless_test_i]]
	D['text_test_inputs'] = [vecUsers[text_test_i], vecItems[text_test_i]]
	D['text_test_outputs'] = [reshapy(vecOutputData[text_test_i]), ratingsData[text_test_i]]

	return D

def embedding_rating_joint(vecUsers, vecItems, vecOutputData, ratingsData, trainingP, hidden_size=8, epochs=3500,
	activations=["tanh", "tanh"], **kwargs):
	embedding_dimensionality = utils.getEmbeddingDim(vecOutputData)
	vecOutputData = np.asarray(vecOutputData)
	ratingsData = np.asarray(ratingsData)
	model, model_textless = models.create_joint_model(vecUsers.shape[1], vecItems.shape[1], hidden_size, embedding_dimensionality, activations=activations, **kwargs)

	D = organize_data(vecUsers, vecItems, vecOutputData, ratingsData, trainingP)
	
	model.summary()

	early_stopping = EarlyStopping(monitor='loss', patience=4)
	early_stopping_val = EarlyStopping(monitor='val_loss', patience=8)
	batch_size = BATCH_SIZE
	model.fit(D['text_train_inputs'], D['text_train_outputs'], validation_split=0.2, callbacks=[early_stopping, early_stopping_val], batch_size=batch_size, epochs=epochs)
	
	evaluation = evaluate(model, D['text_train_inputs'], D['text_train_outputs'], D['text_test_inputs'], D['text_test_outputs'], output_i=1, csvname="embjoint")
	
	user_hidden = diagnostics.get_layer_output(model, ["user_1hot", "item_1hot"], "user_hidden", D['text_train_inputs'])
	item_hidden = diagnostics.get_layer_output(model, ["user_1hot", "item_1hot"], "item_hidden", D['text_train_inputs'])
	features = diagnostics.get_layer_output(model, ["user_1hot", "item_1hot"], "features", D['text_train_inputs'])
	# for i in range(len(features)):
	# 	print(str(user_hidden[i]) + " * " + str(item_hidden[i]))
	# 	print(features[i])
	return evaluation, user_hidden, item_hidden, features

def matrix_factorization(vecUsers, vecItems, ratingsData, trainingP, hidden_size=4, epochs=3500, **kwargs):
	trainingN = int(len(ratingsData) * trainingP) if type(trainingP) is float else trainingP

	train_inputs = [vecUsers[:trainingN], vecItems[:trainingN]]
	train_outputs = ratingsData[:trainingN]
	test_inputs = [vecUsers[trainingN:], vecItems[trainingN:]]
	test_outputs = ratingsData[trainingN:]

	nzUsers_train = np.nonzero(train_inputs[0])[1]
	nzUsers_test = np.nonzero(test_inputs[0])[1]
	trainUsers = set(nzUsers_train)
	testUsers = set(nzUsers_test)
	pctTestDataAreUsersSeenInTraining = np.mean([u in trainUsers for u in nzUsers_test])
	print("pctTestDataAreUsersSeenInTraining " + str(pctTestDataAreUsersSeenInTraining))

	nzItems_train = np.nonzero(train_inputs[1])[1]
	nzItems_test = np.nonzero(test_inputs[1])[1]
	trainItems = set(nzItems_train)
	testItems = set(nzItems_test)
	pctTestDataAreItemsSeenInTraining = np.mean([u in trainItems for u in nzItems_test])
	print("pctTestDataAreItemsSeenInTraining " + str(pctTestDataAreItemsSeenInTraining))

	model = models.create_factorization_model(vecUsers.shape[1], vecItems.shape[1], hidden_size, **kwargs)
	model.summary()
	result = matrix_factorization_simple(model, train_inputs, train_outputs, test_inputs, test_outputs, epochs)

	if hidden_size:
		user_hidden = diagnostics.get_layer_output(model, ["user_1hot", "item_1hot"], "user_hidden", train_inputs)
		item_hidden = diagnostics.get_layer_output(model, ["user_1hot", "item_1hot"], "item_hidden", train_inputs)
		for i in range(min(100,len(user_hidden))):
			print(str(user_hidden[i]) + "	" + str(item_hidden[i]))

	return result

def matrix_factorization_simple(model, train_inputs, train_outputs, test_inputs, test_outputs, epochs=3500):
	callbacks = [EarlyStopping(monitor='loss', patience=3), EarlyStopping(monitor='val_loss', patience=6)]
	batch_size = BATCH_SIZE
	model.fit(train_inputs, train_outputs, validation_split=0.2, callbacks=callbacks, batch_size=batch_size, epochs=epochs)
	return evaluate(model, train_inputs, train_outputs, test_inputs, test_outputs, csvname="matfact")

def deepconn_1st_order(matUserInputData, matItemInputData, ratingsData, trainingP, u_seq_len=200, i_seq_len=200, hidden_size=4, epochs=3500, derp=False):
	trainingN = int(len(ratingsData) * trainingP) if type(trainingP) is float else trainingP

	print("Setting up data...")
	u_mat_train = [mat.get_np_vec(only_training=True) for mat in matUserInputData[:trainingN]]
	i_mat_train = [mat.get_np_vec(only_training=True) for mat in matItemInputData[:trainingN]]
	u_mat_test = [mat.get_np_vec(only_training=False) for mat in matUserInputData[trainingN:]]
	i_mat_test = [mat.get_np_vec(only_training=False) for mat in matItemInputData[trainingN:]]

	embedding_dimensionality = u_mat_train[0].shape[1]
	deepconn = models.DeepCoNN(embedding_dimensionality, hidden_size, u_seq_len, i_seq_len)
	derpcon = models.DerpCon(embedding_dimensionality, hidden_size, u_seq_len, i_seq_len)

	model = deepconn.create_deepconn_dp() if not derp else derpcon.create_derpcon_dp()

	u_mat_train = keras.preprocessing.sequence.pad_sequences(u_mat_train, maxlen=u_seq_len)
	i_mat_train = keras.preprocessing.sequence.pad_sequences(i_mat_train, maxlen=i_seq_len)
	u_mat_test = keras.preprocessing.sequence.pad_sequences(u_mat_test, maxlen=u_seq_len)
	i_mat_test = keras.preprocessing.sequence.pad_sequences(i_mat_test, maxlen=i_seq_len)

	outputs = np.asarray(ratingsData)
	model.summary()

	train_outputs = outputs[:trainingN]
	test_outputs = outputs[trainingN:]

	early_stopping = EarlyStopping(monitor='loss', patience=4)
	early_stopping_val = EarlyStopping(monitor='val_loss', patience=6)
	batch_size = BATCH_SIZE
	model.fit([u_mat_train, i_mat_train], train_outputs, validation_split=0.2, callbacks=[early_stopping, early_stopping_val], batch_size=batch_size, epochs=epochs)
	return evaluate(model, [u_mat_train, i_mat_train], train_outputs, [u_mat_test, i_mat_test], test_outputs,
		metrics=["baseline_train_mse", "model_train_mse", "baseline_test_mse", "model_test_mse", "model_test_corr"])




