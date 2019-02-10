import utils
import numpy as np
import simulate
import matplotlib.pyplot as plt

SEED = None
MAXRATINGS = 150000

def setup_data_from_simfile():
	rawInputData, ratingsData = utils.initRatingsInputOutputData('simulated.csv', maxlines=MAXRATINGS, save=False)
	all_users = utils.getSetFromData('u', rawInputData)
	all_items = utils.getSetFromData('asin', rawInputData)
	vecInputData, vecOutputData = utils.initVecData(rawInputData, None, None, save=False)
	vecUsers = vecInputData[:,len(all_items):]
	vecItems = vecInputData[:,:len(all_items)]
	nzUsers = np.nonzero(vecUsers)
	nzItems = np.nonzero(vecItems)
	assert(len(nzUsers[0]) == len(vecUsers))
	assert(len(nzItems[0]) == len(vecItems))
	print("density= " + str(len(ratingsData) / (len(all_users) * len(all_items))))
	return vecUsers, vecItems, ratingsData

def generate_real_collected_data(n_ratings, n_users, n_items, u_mu=[0,0,0], u_sig=[.1,1,.5], i_mu=[0,0,0], i_sig=[.1,1.2,.8]):
	simulate.gen_file(n_ratings, n_users, n_items, u_mu, u_sig, i_mu, i_sig, seed=SEED)
	return setup_data_from_simfile()

def generate_real_collected_data_random(n_ratings, n_users, n_items, n_attributes):
	u_mu = np.zeros(n_attributes)
	i_mu = np.zeros(n_attributes)
	u_sig = np.random.lognormal(-1,.7,n_attributes)
	i_sig = np.random.lognormal(-1,.7,n_attributes)
	return generate_real_collected_data(n_ratings, n_users, n_items, u_mu, u_sig, i_mu, i_sig)

def generate_fake_collected_data(n_ratings, n_users, n_items, r_samples):
	simulate.gen_file_fake(n_ratings, n_users, n_items, r_samples)
	return setup_data_from_simfile()

def one_dataset_experiment(uv, iv, r, hidden_size, epochs):
	import models
	from keras.callbacks import EarlyStopping
	matfac_model = models.create_factorization_model(uv.shape[1], iv.shape[1], hidden_size=hidden_size, useIntercepts=True)
	matfac_model.fit([uv, iv], r, callbacks=[EarlyStopping(monitor='loss', patience=3)], batch_size=32, epochs=epochs)
	model_train_mse = matfac_model.evaluate([uv, iv], r, verbose=0)
	return model_train_mse

def run_experiment(n_users=100, n_items=20, n_ratings=1000, sim_hidden_size=3, model_hidden_size=3, epochs=500, smart_fake=False):
	real_u_v, real_i_v, real_r = generate_real_collected_data_random(n_ratings, n_users, n_items, sim_hidden_size)
	fake_dist = real_r if smart_fake else np.random.uniform(min(real_r), max(real_r), len(real_r))
	fake_u_v, fake_i_v, fake_r = generate_fake_collected_data(n_ratings, n_users, n_items, fake_dist)
	real_result = one_dataset_experiment(real_u_v, real_i_v, real_r, model_hidden_size, epochs)
	fake_result = one_dataset_experiment(fake_u_v, fake_i_v, fake_r, model_hidden_size, epochs)
	return {"real_collector_mse":real_result, "fake_collector_mse":fake_result}

n_experiments = 10
results = []
import csv
with open('adversarial_results.csv', 'w') as wcsv:
	csvwriter = csv.writer(wcsv, delimiter=',')
	csvwriter.writerow(["real_collector_mse", "fake_collector_mse"])
	for i in range(n_experiments):
		result = run_experiment(n_users=20, n_items=20, n_ratings=200, sim_hidden_size=3, model_hidden_size=3, epochs=200, smart_fake=True)
		csvwriter.writerow([result["real_collector_mse"], result["fake_collector_mse"]])
		results.append(result)
plt.hist([r["real_collector_mse"] for r in results], alpha=0.5)
plt.hist([r["fake_collector_mse"] for r in results], alpha=0.5)
plt.show()

# other experiments:
# try when simulator's attribute vectors are larger than the model hidden layer (sim_hidden_size > model_hidden_size)
# try with small datasets (n_users=20, n_items=20, n_ratings=200)
