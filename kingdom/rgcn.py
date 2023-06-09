import math
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import spmm
from torch_scatter import scatter
from torch_geometric.utils import softmax, degree
import pdb

embedding = torch.Tensor(pickle.load(open('utils/max_entity_embedding.pkl', 'rb')))
#relation_embedding = pickle.load(open('utils/TRue_relation_embedding.pkl', 'rb'))


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


class RGAT(torch.nn.Module):
    def __init__(self, num_relations, num_bases, dropout, e_hidden=100, r_hidden=100):
        super(RGAT, self).__init__()

        self.entity_embedding = embedding
        self.relation_embedding = nn.Parameter(torch.Tensor(76, 100))
        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))
        self.gcn1 = GCN()
        self.highway1 = Highway(e_hidden)
        self.gcn2 = GCN()
        self.gcn3 = GCN()
        self.highway2 = Highway(e_hidden)
        self.highway2 = Highway(e_hidden)
        self.gat_e_to_r = GAT_E_to_R(e_hidden, r_hidden)
        self.gat_r_to_e = GAT_R_to_E(e_hidden, r_hidden)
        #self.highway3 = Highway(e_hidden)
        self.rel_gat = GraphAttention_wjq(e_hidden, r_hidden)
        self.gat = GAT(100)
        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, rel_all, rel, edge_index_all):
        #values = torch.ones(edge_index.size(1)).to('cuda')
        #a = torch.sparse_coo_tensor(edge_index, values, size=[len(entity),len(entity)])
        x_e = self.entity_embedding[entity.long()].cuda()
        x = x_e
        rel_emb = self.relation_embedding

        #s = self.gcn1(x,edge_index_all)
        x = self.highway1(x, self.gcn1(x, edge_index_all))
        x = self.highway2(x, self.gcn2(x, edge_index_all))
        x_r = self.gat_e_to_r(x, edge_index, rel)
        # y = self.gat_r_to_e(x_e, x_r, edge_index, rel)
        x = torch.mean(torch.stack([x, self.gat_r_to_e(x, x_r, edge_index, rel)]), dim=0)
        #x = self.highway3(x, self.gcn3(x, edge_index_all))
        #x = self.rel_gat(x, edge_index_all, rel_all, rel_emb)
        #x = self.gat(x, edge_index_all)
        x = torch.mean(torch.stack([x, self.gat(x, edge_index_all)]), dim=0)
        # torch.stack延申扩展，然后取平均值降维

        return x



class GraphAttention_wjq(nn.Module):
    def __init__(self, e_hidden, r_hidden):
        super(GraphAttention_wjq, self).__init__()
        self.e_hidden = e_hidden
        self.r_hidden = r_hidden
        # self.ww1 = nn.Parameter(nn.init.sparse(torch.empty(300,1), sparsity=0.1))
        # self.ww2 = nn.Parameter(nn.init.sparse(torch.empty(300,1), sparsity=0.1))
        self.ww1 = nn.Linear(300, 1, bias=False)

    def forward(self, x, edge_index_all, rel_all, rel_emb):
        outputs = []
        e_features = x
        edge_index_i, edge_index_j = edge_index_all
        e_head = e_features[edge_index_i]
        e_tail = e_features[edge_index_j]
        e_rel = rel_emb[rel_all]
        s = torch.cat([e_head, e_rel, e_tail], dim=1)
        att = self.ww1(torch.cat([e_head, e_rel, e_tail], dim=1)).squeeze()
        att = softmax(att, edge_index_i)
        # new_features = new_features.index_add_(0, adj_matrix[0,:], neighs * torch.unsqueeze(att._values(), axis=-1))
        x_e = x.index_add(0,edge_index_i,torch.mean(torch.stack([e_head, e_rel, e_tail]), dim=0) * torch.unsqueeze(att, dim=-1))
        #          reduce='sum')
        #x_e = scatter(torch.cat([e_head, e_rel, e_tail], dim=1) * torch.unsqueeze(att, dim=-1),edge_index_i,dim=0,reduce='sum'

        outputs.append(x_e)
        x = F.relu(torch.cat(outputs, dim=1))
        return x


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        #self.w = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        deg = degree(edge_index_i, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index_j] * deg_inv_sqrt[edge_index_i]
        y = x.size(0)
        x = F.relu(spmm(edge_index[[1, 0]], norm, x.size(0), x.size(0), x))
        # x = self.w(x)
        return x


class Highway(nn.Module):
    def __init__(self, x_hidden):
        super(Highway, self).__init__()
        self.lin = nn.Linear(x_hidden, x_hidden)

    def forward(self, x1, x2):
        gate = torch.sigmoid(self.lin(x1))  # gate为文章所说的权重
        x = torch.mul(gate, x2) + torch.mul(1 - gate, x1)
        return x


class GAT(nn.Module):
    def __init__(self, hidden):
        super(GAT, self).__init__()
        self.a_i = nn.Linear(hidden, 1, bias=False)
        self.a_j = nn.Linear(hidden, 1, bias=False)
        self.a_r = nn.Linear(hidden, 1, bias=False)  # 定义变换矩阵

    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        e_i = self.a_i(x).squeeze()[edge_index_i.long()]  #先把特征映射到实数上面去然后进行增维
        e_j = self.a_j(x).squeeze()[edge_index_j.long()]
        e = e_i + e_j  # 拼接经过线性变换的头尾实体
        alpha = softmax(F.leaky_relu(e).float(), edge_index_i)
        x = F.relu(spmm(edge_index[[1, 0]], alpha, x.size(0), x.size(0), x))


        return x

class GAT_E_to_R(nn.Module):
    def __init__(self, e_hidden, r_hidden):  # 关系和实体的维度
        super(GAT_E_to_R, self).__init__()
        self.a_h1 = nn.Linear(r_hidden, 1, bias=False)  # 头实体
        self.a_h2 = nn.Linear(r_hidden, 1, bias=False)  #
        self.a_t1 = nn.Linear(r_hidden, 1, bias=False)  # 尾实体
        self.a_t2 = nn.Linear(r_hidden, 1, bias=False)
        self.w_h = nn.Linear(e_hidden, r_hidden, bias=False)  # 权重
        self.w_t = nn.Linear(e_hidden, r_hidden, bias=False)

    def forward(self, x_e, edge_index, rel):
        edge_index_h, edge_index_t = edge_index
        x_r_h = self.w_h(x_e)
        x_r_t = self.w_t(x_e)

        e1 = self.a_h1(x_r_h).squeeze()[edge_index_h] + self.a_h2(x_r_t).squeeze()[edge_index_t]
        e2 = self.a_t1(x_r_h).squeeze()[edge_index_h] + self.a_t2(x_r_t).squeeze()[edge_index_t]

        alpha = softmax(F.leaky_relu(e1).float(), rel)
        x_r_h = spmm(torch.cat([rel.view(1, -1), edge_index_h.view(1, -1)], dim=0), alpha, rel.max() + 1, x_e.size(0),
                     x_r_h)

        alpha = softmax(F.leaky_relu(e2).float(), rel)
        x_r_t = spmm(torch.cat([rel.view(1, -1), edge_index_t.view(1, -1)], dim=0), alpha, rel.max() + 1, x_e.size(0),
                     x_r_t)
        x_r = x_r_h + x_r_t
        return x_r


class GAT_R_to_E(nn.Module):
    def __init__(self, e_hidden, r_hidden):
        super(GAT_R_to_E, self).__init__()
        self.a_h = nn.Linear(e_hidden, 1, bias=False)
        self.a_t = nn.Linear(e_hidden, 1, bias=False)
        self.a_r = nn.Linear(r_hidden, 1, bias=False)

    def forward(self, x_e, x_r, edge_index, rel):
        edge_index_h, edge_index_t = edge_index
        e_h = self.a_h(x_e).squeeze()[edge_index_h]
        e_t = self.a_t(x_e).squeeze()[edge_index_t]
        e_r = self.a_r(x_r).squeeze()[rel]
        alpha = softmax(F.leaky_relu(e_h + e_r).float(), edge_index_h)
        x_e_h = spmm(torch.cat([edge_index_h.view(1, -1), rel.view(1, -1)], dim=0), alpha, x_e.size(0), x_r.size(0),
                     x_r)
        alpha = softmax(F.leaky_relu(e_t + e_r).float(), edge_index_t)
        x_e_t = spmm(torch.cat([edge_index_t.view(1, -1), rel.view(1, -1)], dim=0), alpha, x_e.size(0), x_r.size(0),
                     x_r)
        x = torch.mean(torch.stack([x_e_h, x_e_t]), dim=0)
        return x