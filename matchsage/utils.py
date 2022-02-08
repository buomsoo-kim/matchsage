import torch

def construct_negative_graph(graph, k, etype, device):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k).to(device)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,)).to(device)
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}).to(device)

def construct_negative_graph_ms(graph, etype1, etype2, device, num_edges = 1000):
    src, dst = graph.edges(etype=etype1)
    indices = torch.LongTensor(np.random.choice(src.size(0), num_edges)).to(device)
    pos_graph = dgl.heterograph(
        {etype1: (src[indices], dst[indices])},
        num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})
    neg_src, neg_dst = graph.edges(etype=etype2)
    indices = torch.LongTensor(np.random.choice(neg_src.size(0), num_edges)).to(device)
    neg_graph = dgl.heterograph(
        {etype2: (neg_src[indices], neg_dst[indices])},
        num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})
    return pos_graph, neg_graph

def compute_loss(pos_score, neg_score):
    # Margin loss
    return (1 - pos_score + neg_score).clamp(min=0).mean()