import numpy as np
from sklearn.metrics import *

def hit_ratio(ground_truth, unique_items, y_prob, k):
	temp = []
	for i in range(len(unique_items)):
		temp.append((unique_items[i], y_prob[i])) 
	temp = [x[0] for x in sorted(temp, key=lambda l:l[1], reverse=True)[:k]]
	hits = set(ground_truth).intersection(set(temp))
	return len(hits)/len(ground_truth)

def ndcg(ground_truth, unique_items, y_prob, k):
	temp = np.zeros(len(unique_items))
	for i in range(len(unique_items)):
		if unique_items[i] in ground_truth:
			temp[i] = 1
		else:
			temp[i] = 0
	temp, y_prob= temp.reshape(1, -1), np.array(y_prob).reshape(1, -1)
	return ndcg_score(temp, y_prob, k = k)