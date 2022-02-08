import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import *
from models import GCN, GraphSAGE, GAT, MatchSAGE


# import & process data
training_data = pd.read_csv('data/training_data.csv')
train_accepted = train_data[train_data["status"] == 'accept']
train_rejected = train_data[train_data["status"] == 'reject']
unique_users = list(set(pd.concat([training_data["source_id"], training_data["user_id"]])))

rating_data = pd.read_csv('data/rating_data.csv')
rating_data_with_records = rating_data[(rating_data["user_id"].isin(unique_users)) & (rating_data["source_id"].isin(unique_users))]

user_to_idx, idx_to_user = {}, {}
for i in tqdm(range(len(unique_users))):
  uid = unique_users[i]
  user_to_idx[uid] = i
  idx_to_user[i] = uid

# hyperparameter settings
NODE_FEATURE_DIM = 50
HIDDEN_FEATURE_DIM = 100
EMBEDDING_DIM = 100
MLP_HIDDEN_DIM = 30
NUM_EPOCHS = 100
NUM_HEADS = 6 # used only when training GAT
MODEL = 'MATCHSAGE' # specify the model - one of {'GCN', 'GRAPHSAGE', 'GAT', 'MATCHSAGE'}
alpha = .7
k1, k2 = 1000, 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training in device: ", device)


# creating multigraph
torch.manual_seed(777)
acc_src, acc_dst = torch.LongTensor(train_accepted['user_id'].map(user_to_idx).values).to(device), torch.LongTensor(train_accepted['source_id'].map(user_to_idx).values).to(device)
rej_src, rej_dst = torch.LongTensor(train_rejected['user_id'].map(user_to_idx).values).to(device), torch.LongTensor(train_rejected['source_id'].map(user_to_idx).values).to(device)
rate_src, rate_dst = torch.LongTensor(rating_data_with_records["user_id"].map(user_to_idx).values).to(device),torch.LongTensor(rating_data_with_records["source_id"].map(user_to_idx).values).to(device)

graph = dgl.heterograph({
    ('user', 'accepts', 'user'): (acc_src, acc_dst),
    ('user', 'rejects', 'user'): (rej_src, rej_dst),
    ('user', 'rates', 'user'): (rate_src, rate_dst)
},  num_nodes_dict = {"user": len(idx_to_user)}).to(device)

graph.nodes['user'].data['src_feature'] = torch.randn(graph.nodes("user").size(0), NODE_FEATURE_DIM).to(device)
graph.nodes['user'].data['dst_feature'] = torch.randn(graph.nodes("user").size(0), NODE_FEATURE_DIM).to(device)
graph.edges['rates'].data['label'] = torch.tensor(rating_data_with_records["rating"].values).to(device)


# GNN training
if MODEL == 'GCN':
	model = GCN(NODE_FEATURE_DIM, HIDDEN_FEATURE_DIM, EMBEDDING_DIM, graph.etypes).to(device)
elif MODEL == 'GRAPHSAGE':
	model = GraphSAGE(NODE_FEATURE_DIM, HIDDEN_FEATURE_DIM, EMBEDDING_DIM, graph.etypes).to(device)
elif MODEL == 'GAT':
	model = GAT(NODE_FEATURE_DIM, HIDDEN_FEATURE_DIM, EMBEDDING_DIM, graph.etypes, NUM_HEADS).to(device)
elif MODEL == 'MATCHSAGE':
	model = MatchSAGE(NODE_FEATURE_DIM, HIDDEN_FEATURE_DIM, EMBEDDING_DIM, graph.etypes).to(device)


opt = torch.optim.Adam(model.parameters())
src_feats, dst_feats = {"user": graph.nodes['user'].data['src_feature']} , {"user": graph.nodes['user'].data['dst_feature']}
etype_1, etype_2, etype_3 = ('user', 'accepts', 'user'),  ('user', 'rejects', 'user'), ('user', 'rates', 'user')
edge_label = graph.edges['rates'].data['label']

for epoch in range(NUM_EPOCHS):
	if MODEL == 'MATCHSAGE':
    	pos_graph, negative_graph = construct_negative_graph_ms(graph, etype_1, etype_2, device, k2)
    	pos_score, neg_score, rating = model(graph, pos_graph, negative_graph, src_feats, dst_feats, etype_1, etype_2, etype_3)
    	loss_a = compute_loss(pos_score, neg_score)
    	loss_b = ((rating.squeeze() - edge_label) ** 2).mean()
    	loss = alpha * loss_a + (1-alpha) * loss_b
    else:
    	negative_graph = construct_negative_graph(graph, k2, etype_1, device)
    	pos_score, neg_score = model(graph, negative_graph, src_feats, etype_1)
    	loss = compute_loss(pos_score, neg_score)

    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 10 == 0:
      print("Loss at epoch {}: {}".format(epoch+1, loss.item()))


# retrieve and export embeddings 
embeddings_src, embeddings_dst = model.sage(graph, src_feats)['user'].detach().cpu().numpy(), model.sage(graph, dst_feats)['user'].detach().cpu().numpy()
embeddings_src, embeddings_dst = pd.DataFrame(embeddings_src, index = [x[1] for x in idx_to_user.items()]), pd.DataFrame(embeddings_dst, index = [x[1] for x in idx_to_user.items()])

src_to_embedding, dst_to_embedding = dict(), dict()

for i in tqdm(range(len(embeddings_src))):
  src_to_embedding[embeddings_src.index[i]] = embeddings_src.iloc[i].values
  dst_to_embedding[embeddings_dst.index[i]] = embeddings_dst.iloc[i].values

if MODEL == 'MATCHSAGE':
	embeddings_src.to_csv('active_embeddings.csv')
	embeddings_dst.to_csv('passive_embeddings.csv')
else:
	embeddings_src.to_csv('embeddings.csv')