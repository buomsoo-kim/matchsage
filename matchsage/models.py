import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl.function as fn
import dgl

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats=in_feats, out_feats=hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats=hid_feats, out_feats=out_feats)
            for rel in rel_names}, aggregate='sum')

class RSAGEConv(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

class RGATConv(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, num_heads):
        super().__init__()
        self.hid_feats = hid_feats
        self.num_heads = num_heads
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_feats, hid_feats, num_heads)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(hid_feats * num_heads, out_feats, num_heads)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v).view(-1, self.hid_feats * self.num_heads) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: h[k].mean(axis=1) for k, v in h.items()}
        return h

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()
    def forward(self, g, neg_g, x, etype):
        h = self.sage(g, x)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)

class GraphSAGE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = RSAGEConv(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()
    def forward(self, g, neg_g, x, etype):
        h = self.sage(g, x)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)

class GAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names, num_heads):
        super().__init__()
        self.sage = RGATConv(in_features, hidden_features, out_features, rel_names, num_heads)
        self.pred = HeteroDotProductPredictor()
    def forward(self, g, neg_g, x, etype):
        h = self.sage(g, x)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)

class MatchSAGE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = RSAGEConv(in_features, hidden_features, out_features, rel_names)
        self.link_prediction = HeteroDotProductPredictor()
        self.mlp_rating_prediction = MLPPredictor(out_features, 30, 1)
    def forward(self, g, pos_g, neg_g, x1, x2, etype1, etype2, etype3):
        h_src = self.sage(g, x1)     
        h_dst = self.sage(g, x2)     
        return self.link_prediction(pos_g, h_src, h_dst, etype1), self.link_prediction(neg_g, h_src, h_dst, etype2),  self.mlp_rating_prediction(g, h_src, h_dst, etype3)

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h_src, h_dst, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h_src'] = h_src['user']
            graph.ndata['h_dst'] = h_dst['user']
            graph.apply_edges(fn.u_dot_v('h_src', 'h_dst', 'score'), etype=etype)
            return graph.edges[etype].data['score']

class MLPPredictor(nn.Module):
    def __init__(self, in_features, hidden_dim, out_classes):
        super().__init__()
        self.W1 = nn.Linear(in_features, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.W2 = nn.Linear(hidden_dim, out_classes)
        self.relu = nn.ReLU()

    def apply_edges(self, edges):
        h_u = edges.src['h_src']
        h_v = edges.dst['h_dst']
        x = self.W1(h_u * h_v)
        x = self.bn(x)
        x = self.relu(x)
        score = self.W2(x)
        return {'score': score}

    def forward(self, graph, h_src, h_dst, etype):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h_src'] = h_src['user']
            graph.ndata['h_dst'] = h_dst['user']           
            graph.apply_edges(self.apply_edges, etype=etype)
            return graph.edges[etype].data['score']