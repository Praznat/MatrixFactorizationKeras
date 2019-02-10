import numpy as np
import csv
import json
import matplotlib.pyplot as plt

def generate_multimodal_normals(mus, sigs, proportions, n):
	''' mus and sigs both CxH matrices, C being number of clusters and H being dimensionality of resulting draws '''
	d = len(mus)
	assert(d == len(proportions))
	assert(len(sigs) == len(proportions))
	assert(len(mus[0]) == len(sigs[0]))
	x = np.zeros((n, len(mus[0])))
	c = np.zeros(n, dtype=int)
	proportions /= np.sum(proportions)
	ds = range(d)
	for i in range(n):
		c[i] = np.random.choice(ds, p=proportions)
		mu = mus[c[i]]
		sig = sigs[c[i]]
		x[i] = np.random.normal(mu, sig)
	return c, x

def generate_demographic_matfac_simdata(n_ratings, n_users, n_items, d_mu_mu, d_mu_sig, d_sig_max, n_demographics, i_mu, i_sig, min_cut = 1.8, max_cut = 0.8, seed=None):
	if seed is not None:
		np.random.seed(seed)
	assert(len(d_mu_mu)==len(d_mu_sig))
	u_mus = np.random.normal(d_mu_mu, d_mu_sig, (n_demographics, len(d_mu_mu)))
	u_sigs = np.random.uniform(0, d_sig_max, (n_demographics, len(d_mu_mu))) # demographic attribute stdevs drawn from uniform...
	u_proportions = np.random.dirichlet(np.ones(n_demographics))
	print("demographic centers: ")
	print(u_mus)
	print("demographic stdevs: ")
	print(u_sigs)
	print("demographic proportions: ")
	print(u_proportions)
	c, u_v = generate_multimodal_normals(u_mus, u_sigs, u_proportions, n_users)
	i_v = np.random.normal(i_mu, i_sig, (n_items, len(i_mu)))
	results = generate_matrix_factorization_simdata_from_attributes(n_ratings, n_users, n_items, u_v, i_v, min_cut, max_cut, seed)
	return {'results':results, 'demographics':c}

def generate_matrix_factorization_simdata(n_ratings, n_users, n_items, u_mu, u_sig, i_mu, i_sig, min_cut = 1.8, max_cut = 0.8, seed=None):
	if seed is not None:
		np.random.seed(seed)
	u_v = np.random.normal(u_mu, u_sig, (n_users, len(u_mu)))
	i_v = np.random.normal(i_mu, i_sig, (n_items, len(i_mu)))
	return generate_matrix_factorization_simdata_from_attributes(n_ratings, n_users, n_items, u_v, i_v, min_cut, max_cut, seed)

def random_select_users_items(n_ratings, n_users, n_items):
	k_u_ = np.random.choice(range(n_users), size=n_ratings*2)
	k_i_ = np.random.choice(range(n_items), size=n_ratings*2)
	existing = set()
	k_u = []
	k_i = []
	for k in range(n_ratings*2):
		newone = (k_u_[k], k_i_[k])
		if newone not in existing:
			k_u.append(k_u_[k])
			k_i.append(k_i_[k])
		existing.add(newone)
	return k_u[:n_ratings], k_i[:n_ratings]

def generate_matrix_factorization_simdata_from_attributes(n_ratings, n_users, n_items, u_v, i_v, min_cut, max_cut, seed):
	# plt.hist(u_v)
	# plt.show()
	result = []
	k_u, k_i = random_select_users_items(n_ratings, n_users, n_items)
	r_u_v = u_v[k_u]
	r_i_v = i_v[k_i]
	r = r_u_v[:,0] + r_i_v[:,0] + np.sum(r_u_v[:,1:] * r_i_v[:,1:], axis=1) # so no noise?

	r_mu = np.mean(r)
	r_sd = np.std(r)
	rmin = r_mu - min_cut * r_sd
	rmax = r_mu + max_cut * r_sd
	r = np.minimum(np.maximum(r, rmin), rmax)
	a = .1
	r = (r - rmin) / (rmax - rmin) + a
	return {"users":k_u, "items":k_i, "ratings":r}

def write_files(k_u, k_i, r, filename="simulated"):
	with open(filename + '.csv', 'w', newline='') as csvf:
		csvWriter = csv.writer(csvf, delimiter=',')
		for i in range(len(r)):
			csvWriter.writerow(["u" + str(k_u[i]), "i" + str(k_i[i]), r[i], "1900"])
	with open(filename + '.json', 'w', newline='') as jsonf:
		for i in range(len(r)):
			text = "version control is great"
			jstring = json.dumps({'reviewerID': "u" + str(k_u[i]), 'asin': "i" + str(k_i[i]), 'reviewText': text})
			jsonf.write(jstring + '\n')

def gen_file(n_ratings, n_users, n_items, u_mu, u_sig, i_mu, i_sig, seed=None):
	sim = generate_matrix_factorization_simdata(n_ratings, n_users, n_items, u_mu, u_sig, i_mu, i_sig, seed=seed)
	write_files(sim['users'], sim['items'], sim['ratings'])

def gen_file_demographic(n_ratings, n_users, n_items, d_mu_mu, d_mu_sig, d_sig_max, n_demographics, i_mu, i_sig, seed=None):
	simD = generate_demographic_matfac_simdata(n_ratings, n_users, n_items, d_mu_mu, d_mu_sig, d_sig_max, n_demographics, i_mu, i_sig, seed=seed)
	sim = simD['results']
	write_files(sim['users'], sim['items'], sim['ratings'])
	return simD['demographics']

def gen_file_fake(n_ratings, n_users, n_items, r_samples):
	''' use r_samples to populate fake ratings. can be drawn from random distribution or use real data from somewhere else '''
	k_u, k_i = random_select_users_items(n_ratings, n_users, n_items)
	write_files(k_u, k_i, r_samples[:n_ratings])

# gen_file(100, 20, 10, [0,0,0],[1, 1, .5], [0,0,0], [1, 1.2, .8])