import utils
import sys
import numpy as np
import diagnostics
import experiments
import simulate
import argparse

parser = argparse.ArgumentParser()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument('--seed', default=None)
parser.add_argument('--reviews_file', default='reviews_Video_Games_5.json')
parser.add_argument('--ratings_file', default='ratings_Video_Games.csv')
parser.add_argument('--useSimulator', default=False, type=str2bool) # overrides given reviews and ratings files
parser.add_argument('--maxreviews', default=5000)
parser.add_argument('--epochs', default=5000)
parser.add_argument('--emb_likelihood_wgt', default=1.0)
parser.add_argument('--matFac', default=True, type=str2bool)
parser.add_argument('--matFacText', default=True, type=str2bool)
parser.add_argument('--deepCoNN', default=False, type=str2bool)
parser.add_argument('--textAnalysis', default=True, type=str2bool) # only works if using matFactText
parser.add_argument('--textAnalysisLimitK', default=100) # only works if using textAnalysis
args = parser.parse_args()

if args.seed is not None:
	np.random.seed(args.seed)

reviews_file = args.reviews_file
ratings_file = args.ratings_file
if args.useSimulator:
	simulate.gen_file(500, 50, 20, [0,0,0], [.1, 1, .5], [0,0,0], [.1, 1.2, .8], seed=args.seed)
	reviews_file = 'simulated.json'
	ratings_file = 'simulated.csv'

maxratings = args.maxreviews * 100
epochs = args.epochs

embedding_map = utils.initEmbeddingMap()

st = utils.SparcityTarget(density_target=.02)
rawInputData, rawOutputData = utils.initRawData(input_file=reviews_file, maxlines=args.maxreviews, sparcity_target=st, fileName="dc_raw.p", save=False)

users, items = utils.getSparcityInfo(rawInputData)
print([len(u) for u in users.values()])
print([len(i) for i in items.values()])
extra = {}

rinds = np.random.permutation(len(rawInputData))
rawInputData = [rawInputData[i] for i in rinds]
rawOutputData = [rawOutputData[i] for i in rinds]

ratingsData = utils.initRatingsOutputData(rawInputData, input_file=ratings_file, maxlines=maxratings, fileName="dc_rat.p", save=False)
ratingsData = [r / 5.0 for r in ratingsData] # normalize between 0 and 1 (for ratings between 0 and 5)

all_users = utils.getSetFromData('u', rawInputData)
all_items = utils.getSetFromData('asin', rawInputData)
vecInputData, vecOutputData = utils.initVecData(rawInputData, rawOutputData, embedding_map, save=False)
vecUsers = vecInputData[:,len(all_items):]
vecItems = vecInputData[:,:len(all_items)]
nzUsers = np.nonzero(vecUsers)
nzItems = np.nonzero(vecItems)
assert(len(nzUsers[0]) == len(vecUsers))
assert(len(nzItems[0]) == len(vecItems))
print("density= " + str(len(ratingsData) / (len(all_users) * len(all_items))))

# diagnostics.display_useritem_matrix(vecUsers, vecItems)

trainingP = 0.8

embedding_scaling = 1.0 / utils.getEmbeddingDim(vecOutputData) * args.emb_likelihood_wgt
vecOutputData = [v * embedding_scaling if v is not None else None for v in vecOutputData]
vecOutputData = np.multiply(vecOutputData, embedding_scaling)

bl = experiments.matrix_factorization(vecUsers, vecItems, ratingsData, trainingP=trainingP, hidden_size=0, epochs=epochs, useIntercepts=True)
results = ["Baseline:", bl]

if args.matFac:
	mf = experiments.matrix_factorization(vecUsers, vecItems, ratingsData, trainingP=trainingP, hidden_size=3, epochs=epochs, useIntercepts=True)
	results.append("Matrix factorization:")
	results.append(mf)
if args.matFacText:
	mft, ua, ia, fa = experiments.embedding_rating_joint(vecUsers, vecItems, vecOutputData, ratingsData, trainingP=trainingP, hidden_size=3, epochs=epochs, more_complex=False, useIntercepts=True, activations=["relu","linear"])
	results.append("Matrix factorization with text likelihood regularization:")
	results.append(mft)
if args.deepCoNN:
	matUserInputData, matItemInputData = utils.initMatInputData(rawInputData, rawOutputData, embedding_map, trainingP, fileName="dc_mat.p", save=False, extra_info=extra)
	ptile_max_seq = 50
	u_seq_len = int(np.percentile(np.array(extra['user_seq_sizes']), ptile_max_seq))
	i_seq_len = int(np.percentile(np.array(extra['item_seq_sizes']), ptile_max_seq))
	print(u_seq_len)
	print(i_seq_len)
	dc = experiments.deepconn_1st_order(matUserInputData, matItemInputData, ratingsData, trainingP=trainingP, u_seq_len=u_seq_len, i_seq_len=i_seq_len, derp=False)
	results.append("Matrix factorization with text likelihood regularization:")
	results.append(dc)

for result in results:
	print(result)

print("Finished! (ignore any errors reported after this)")
print()

def save_text_analysis(analysis, name, limit_k):
	num_dims = analysis.shape[1]
	columns = []
	for i in range(num_dims):
		word_scores = diagnostics.word_analysis(analysis[:,i], rawOutputData, 5)
		word_scores = sorted(word_scores, key=word_scores.get)
		columns.append(['top ' + str(limit_k) + ' attribute ' + str(i+1)] + word_scores[:limit_k])
		columns.append(['bottom ' + str(limit_k) + ' attribute ' + str(i+1)] + word_scores[-limit_k:])
	utils.toCsv(np.transpose(np.array(columns)), "attribute_analysis_" + name)

if args.textAnalysis and args.matFacText:
	save_text_analysis(ua, "users", args.textAnalysisLimitK)
	save_text_analysis(ia, "items", args.textAnalysisLimitK)
	save_text_analysis(fa, "interaction", args.textAnalysisLimitK)

