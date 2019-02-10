import active_learning
import utils
import sys
import numpy as np
import simulate
import matplotlib.pyplot as plt

SEED = None #1

n_demographics = 3
n_users = 100
n_items = 20
n_ratings = 1000

# simulate.gen_file(n_ratings, n_users, n_items, [0,0,0], [.1, 1, .5], [0,0,0], [.1, 1.2, .8], seed=SEED)
demographics = simulate.gen_file_demographic(n_ratings, n_users, n_items, [.5,.5,.5],[.3,.3,.3],.8,n_demographics, [0,0,0],[.1, 1.2, .8], seed=SEED)

# ratings_file = 'ratings_ml-20m.csv'
ratings_file = 'simulated.csv'

maxratings = 150000

rawInputData, ratingsData = utils.initRatingsInputOutputData(ratings_file, maxlines=maxratings, save=False)
userdict, itemdict = active_learning.organize_ratings_by_user_item(rawInputData, ratingsData)
# print(userdict)

all_users = list(set([d['u'] for d in rawInputData]))
all_items = list(set([d['asin'] for d in rawInputData]))
all_users_v = utils.to_vec_data([{'u':u} for u in all_users])
all_items_v = utils.to_vec_data([{'i':i} for i in all_items])

print(all_items)
print(all_items_v.shape)

users_to_v = {}
items_to_v = {}

for i in range(len(all_users)):
    users_to_v[all_users[i]] = all_users_v[i]
for i in range(len(all_items)):
    items_to_v[all_items[i]] = all_items_v[i]

pct_train = 0.2
pct_candidacy = 0.4
# pct_train = 0.7
# pct_candidacy = 0.0
userdata = {}
train_data = []
test_data = []
for key, val in userdict.items():
    udata = active_learning.get_train_candidacy_test(val, pct_train, pct_candidacy)
    train_data += udata['TRAIN']
    test_data += udata['TEST']
    userdata[key] = udata

def setup_data_dict(data):
    result = {}
    result['users'] = [d['user'] for d in data]
    result['items'] = [d['item'] for d in data]
    result['ratings'] = [d['rating'] for d in data]
    result['users_v'] = np.array([users_to_v[d['user']] for d in data])
    result['items_v'] = np.array([items_to_v[d['item']] for d in data])
    result['input'] = [result['users_v'], result['items_v']]
    return result

train_dict = setup_data_dict(train_data)
test_dict = setup_data_dict(test_data)

import models
import experiments

def run_active_learning_experiment(model, epochs, active_learner, epochsAL, min_updates=5):
    train_data_l = train_data
    train_dict_l = train_dict
    results = []
    for t in range(epochsAL):
        rs = experiments.matrix_factorization_simple(model, train_dict_l['input'], train_dict_l['ratings'], test_dict['input'], test_dict['ratings'], epochs=epochs)
        results.append(rs)

        new_train_data = []
        updates = 0
        for user, v in userdata.items():
            candidates = v['CANDIDATES']
            if len(v['CANDIDATES']):
                selection = meveAL.get_best_candidate(model, users_to_v[user], items_to_v, candidates)
                new_train_data.append(selection)
                updates += 1

        if updates < min_updates:
            break

        train_data_l += new_train_data
        train_dict_l = setup_data_dict(train_data)
        for d in new_train_data:
            meveAL.add_training_point(d['item'], d['rating'])
    return results

epochs = 5000
epochsAL = 10
min_updates = 5
model_rand = models.create_factorization_model(n_users, n_items, hidden_size=3, useIntercepts=True)
model_meve = models.create_factorization_model(n_users, n_items, hidden_size=3, useIntercepts=True)
randAL = active_learning.RandomAL(train_data)
meveAL = active_learning.MaxErrVsEmpiricalAL(train_data)

rand_experiment = run_active_learning_experiment(model_rand, epochs, randAL, epochsAL)
meve_experiment = run_active_learning_experiment(model_meve, epochs, meveAL, epochsAL)

randplt, = plt.plot([d['model_test_corr'] for d in rand_experiment], label="random selection")
meveplt, = plt.plot([d['model_test_corr'] for d in meve_experiment], color="red", label="active selection")
plt.xlabel("Training labels per user")
plt.ylabel("Test correlation")
plt.legend(handles=[randplt, meveplt], loc=2)
plt.show()