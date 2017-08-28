import utils
import sys
import numpy as np

get_arg = lambda x: sys.argv[x] if x < len(sys.argv) else None

reviews_file = get_arg(1) or 'reviews_Video_Games_5.json' #'sanity_reviews.json'
ratings_file = get_arg(2) or 'ratings_Video_Games.csv' #'sanity_ratings.csv'
maxreviews = get_arg(3) or 5000
maxratings = get_arg(4) or 50000
epochs = get_arg(5) or 150

embedding_map = utils.initEmbeddingMap()

st = utils.SparcityTarget(n_reviews=4)
rawInputData, rawOutputData = utils.initRawData(input_file=reviews_file, maxlines=maxreviews, sparcity_target=st, fileName="dc_raw.p", save=False)
users, items = utils.getSparcityInfo(rawInputData)
print([len(u) for u in users.values()])
print([len(i) for i in items.values()])
extra = {}

rinds = np.random.permutation(len(rawOutputData))
rawInputData = [rawInputData[i] for i in rinds]
rawOutputData = [rawOutputData[i] for i in rinds]

all_users = utils.getSetFromData('u', rawInputData)
all_items = utils.getSetFromData('asin', rawInputData)
vecInputData, vecOutputData = utils.initVecData(rawInputData, rawOutputData, embedding_map, save=False)
vecUsers = vecInputData[:,len(all_users):]
vecItems = vecInputData[:,:len(all_items)]

matUserInputData, matItemInputData = utils.initMatInputData(rawInputData, rawOutputData, embedding_map, fileName="dc_mat.p", save=False, extra_info=extra)
ratingsData = utils.initRatingsOutputData(rawInputData, input_file=ratings_file, maxlines=maxratings, fileName="dc_rat.p", save=False)
ptile = 50
u_seq_len = int(np.percentile(np.array(extra['user_seq_sizes']), ptile))
i_seq_len = int(np.percentile(np.array(extra['item_seq_sizes']), ptile))
print(u_seq_len)
print(i_seq_len)

import experiments

experiments.matrix_factorization(vecUsers, vecItems, ratingsData, epochs=epochs, training=.8)
experiments.deepconn_1st_order(matUserInputData, matItemInputData, ratingsData, u_seq_len=u_seq_len, i_seq_len=i_seq_len, )

