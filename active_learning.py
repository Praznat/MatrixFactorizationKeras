import numpy as np

def organize_ratings_by_user_item(rawInputData, ratingsData):
	if len(rawInputData) != len(ratingsData):
		raise ValueError("same len pls")
	userdict = {}
	itemdict = {}
	for i in range(len(ratingsData)):
		indata = rawInputData[i]
		user = indata['u']
		item = indata['asin']
		rating = ratingsData[i]
		entry = {'user':user, 'item':item, 'rating':rating}
		userdict.setdefault(user, []).append(entry)
		itemdict.setdefault(item, []).append(entry)
	return userdict, itemdict

def get_train_candidacy_test(user_responses, pct_train, pct_candidacy):
	n = len(user_responses)
	n_train = int(round(n * pct_train))
	n_candidacy = int(round(n * pct_candidacy))
	n_test = n - n_train - n_candidacy
	train = user_responses[:n_train]
	candidates = user_responses[n_train:(n_train+n_candidacy)]
	test = user_responses[(n_train+n_candidacy):]
	return {'TRAIN':train, 'CANDIDATES':candidates, 'TEST':test}

def get_predictions(model, users_v, items_v):
    import diagnostics
    return diagnostics.get_layer_output(model, ['user_1hot', 'item_1hot'], "prediction", [users_v, items_v]).flatten()

class ActiveLearningSelector():
	def __init__(self, train_data):
		self.item_to_rating_dist = {}
		for d in train_data:
			self.add_training_point(d['item'], d['rating'])
	def add_training_point(self, item, rating):
		self.item_to_rating_dist.setdefault(item, []).append(rating)
	def get_best_candidate(self, model, user_v, items_to_v, item_candidates):
		raise NotImplementedError()

class RandomAL(ActiveLearningSelector):
	def get_best_candidate(self, model, user_v, items_to_v, item_candidates):
		return np.random.choice(item_candidates)

class MaxErrVsEmpiricalAL(ActiveLearningSelector):
	def get_best_candidate(self, model, user_v, items_to_v, item_candidates):
		items_v = np.array([items_to_v[c['item']] for c in item_candidates])
		user_v_tiled = np.tile(user_v, (len(items_v), 1))
		predictions = get_predictions(model, user_v_tiled, items_v)
		expected_errors = []
		for c in range(len(item_candidates)):
			item = item_candidates[c]['item']
			prediction = predictions[c]
			rating_dist = self.item_to_rating_dist[item]
			expected_errors.append(np.sqrt(np.mean((rating_dist - prediction)**2)))
		return item_candidates[np.argmax(expected_errors)]



