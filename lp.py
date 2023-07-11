from util import *
from graph import *

T = 2000

def get_new_link(cur_edgelist_file):
    graph_data = get_data(cur_edgelist_file)[1]
    num_nodes = graph_data.x.size()[1]
    w = graph_data.edge_weight/ torch.max(graph_data.edge_weight)
    G, F = graph_data.x
    E, W = [], []
    t = 0

    # get edges from edge_index
    edge_dict = {}
    for i in range(num_nodes):
        cur_list = []
        ids = find_indices(list(graph_data.edge_index[0]), 0)
        for id in ids:
            cur_list.append((list(graph_data.edge_index[1])[id], list(w)[id]))
        edge_dict[i] = cur_list

    while (True):
        new_G = G
        new_F = F
        for i in range(num_nodes):
            cur_neigh = len(edge_dict[i])
            cur_G = 0
            cur_F = 0
            for cur_list in edge_dict[i]:
                cur_G = torch.add(cur_G, torch.mul(F[torch.Tensor.int(cur_list[0])], cur_list[1]))
                cur_F = torch.add(cur_F, torch.abs(torch.add(torch.neg(G[torch.Tensor.int(cur_list[0])])), cur_list[1]))
        
            cur_G = torch.div(cur_G, cur_neigh)
            cur_F /= torch.div(cur_F, 2*cur_neigh)
            new_G[i] = cur_G
            new_F[i] = 1-cur_F
            G = new_G
            F = new_F
        t += 1
        if torch.sum(torch.abs(F-new_F))<=1e-3 or torch.sum(torch.abs(G-new_G))<=1e-3 or t>=T:
            break

    E_list = []
    W_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            cur_w = torch.max(w)*(torch.dot(F[i], G[j])+torch.dot(F[j], G[i]))/2
            if cur_w>30:
                E_list.append([i, j])
                E_list.append([j, i])
                W_list.append(cur_w)

    return E_list, W_list

if __name__ == "__main__":
    cur_edgelist_file = "output/GF/raw_data/GF.edgelist"
    print(get_new_link(cur_edgelist_file))