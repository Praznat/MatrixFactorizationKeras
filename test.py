import utils
import numpy as np

embedding_map = utils.initEmbeddingMap()

rawInputData, rawOutputData = utils.initRawData(input_file='sanity_reviews.json', maxlines=500, save=False)
all_users = utils.getSetFromData('u', rawInputData)
all_items = utils.getSetFromData('asin', rawInputData)

vecInputData, vecOutputData = utils.initVecData(rawInputData, rawOutputData, embedding_map, save=False)
vecUsers = vecInputData[:,len(all_items):]
vecItems = vecInputData[:,:len(all_items)]
catItems = np.argmax(vecItems, axis=1)

matUserInputData, matItemInputData = utils.initMatInputData(rawInputData, rawOutputData, embedding_map)
ratingsData = utils.initRatingsOutputData(rawInputData, input_file='sanity_ratings.csv', maxlines=500, save=False)

import models
import experiments

# experiments.user_item_to_embedding(vecInputData, vecOutputData)
# experiments.natlang_u_agnostic(vecItems, vecOutputData, sanity=False)
experiments.natlang_dawidskene(vecUsers, vecItems, vecOutputData)

# experiments.deepconn_1st_order(matUserInputData, matItemInputData, ratingsData, epochs=500)