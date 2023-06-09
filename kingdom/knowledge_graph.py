import pickle
import numpy as np
import torch
from torch import nn
from utils import get_domain_dataset, spacy_seed_concepts_list
from tqdm import  tqdm
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
from torch_geometric.data import Data
embedding = torch.Tensor(pickle.load(open('utils/max_entity_embedding.pkl', 'rb')))
class RGAT(torch.nn.Module):
    def __init__(self):
        super(RGAT, self).__init__()
        self.entity_embedding = embedding
        self.gcn1 = GCN()
        self.highway1 = Highway(100)
    def forward(self, entity,edge_index_all):
        x_e = self.entity_embedding[entity.long()].cuda()
        x = self.highway1(x_e, self.gcn1(x_e, edge_index_all))
        #x = self.gat(x_e, edge_index_all)
        # torch.stack延申扩展，然后取平均值降维
        return x
class GAT(nn.Module):
    def __init__(self,hidden):
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

def generate_graph(xg,share_entity):
    sub_share_entity = sub_share_neighbor(xg, share_entity)
    src, rel, dst = xg.transpose()
    src = torch.tensor(src, dtype=torch.long).contiguous().cuda()
    dst = torch.tensor(dst, dtype=torch.long).contiguous().cuda()
    edge_index = torch.stack((src, dst))
    edge_index_w = share_edge(edge_index, sub_share_entity)
    sg = add_neighbor(xg,edge_index_w)
    return sg

def add_neighbor(uniq_entity,edge_index):

    uniq_entity = uniq_entity.tolist()
    edge_index_i , edge_index_j = edge_index
    edge_index_i = edge_index_i.tolist()
    edge_index_j =  edge_index_j.tolist()
    for i in edge_index_i:
        edge_index_i[edge_index_i.index(i)] = uniq_entity.index(i)
    for j in edge_index_j :
        edge_index_j[edge_index_j.index(j)] = uniq_entity.index(j)
    src = torch.tensor(edge_index_i, dtype=torch.long).contiguous()
    dst = torch.tensor(edge_index_j, dtype=torch.long).contiguous()
    edge_index = torch.stack((src, dst))
    edge_index_all = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    '''
    uniq_entity = np.array(uniq_entity)
    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_index_all = edge_index_all
    '''
    return edge_index_all
def sub_share_neighbor(triplets,entity):
    sub_entity = []
    edges = triplets
    src, rel, dst = edges.transpose()
    uniq_entity = np.unique((src, dst))
    sub_entity.extend(set(uniq_entity.tolist()).intersection(entity))
    return sub_entity

def share_edge(edge_index_s,share_neighbor):
    edge_index_i = []
    edge_index_j = []
    x , y = edge_index_s
    x = x.tolist()
    y = y.tolist()
    for i in share_neighbor:
      if (i in x):
        index = x.index(i)
        edge_index_i.append(i)
        edge_index_j.append(y[index])
      elif(i in y):
          index = y.index(i)
          edge_index_i.append(x[index])
          edge_index_j.append(i)
    src = torch.tensor(edge_index_i, dtype=torch.long).contiguous()
    dst = torch.tensor(edge_index_j, dtype=torch.long).contiguous()
    edge_index = torch.stack((src, dst))
    return edge_index





def unique_rows(a):#删除重复的边
    """
    Drops duplicate rows from a numpy 2d array
    """
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def knowledge_graph(model,domain, split, all_seeds, concept_graphs,entity,edge_index_I,edge_index_J):
    """
        Graph features for each sentence (document) instance in a domain.
    """
    ill = 0
    x, dico = get_domain_dataset(domain, exp_type=split)
    d = list(dico.values())
    sent_features = np.zeros((len(x),100))
    kk = np.zeros((1,100))
    for j in tqdm(range(len(x)), position=0, leave=False):
        uniq_entity = np.array(entity[j])
        edge_index_i = np.array(edge_index_I[j])
        edge_index_j = np.array(edge_index_J[j])
        if (len(uniq_entity) != 0):
            src = torch.tensor(edge_index_i, dtype=torch.long).contiguous()
            dst = torch.tensor(edge_index_j, dtype=torch.long).contiguous()
            edge_index = torch.stack((src, dst))
            data = Data(edge_index=edge_index)
            data.entity = torch.from_numpy(uniq_entity)
            data = data.cuda()
            features = model(data.entity, data.edge_index)
            sent_features[j] = features.cpu().detach().numpy().mean(axis=0)
        else:
            sent_features[j] = kk
        torch.cuda.empty_cache()

        '''
        c = [dico.id2token[item] for item in np.where(x[j] != 0)[0]]
        n = list(spacy_seed_concepts_list(c).intersection(set(all_seeds)))
        try:
            ill  = ill + 1
            xg = np.concatenate([concept_graphs[item] for item in n])  # 内存链接在一起
            xg = xg[~np.all(xg == 0, axis=1)]
            xg = unique_rows(xg).astype('int64')
            if len(xg) > 50000:
                xg = xg[:50000, :]
            sub_share_entity = sub_share_neighbor(xg, share_entity)
            edges = xg
            src, rel, dst = edges.transpose()
            uniq_entity = np.unique((src, dst))
            src = torch.tensor(src, dtype=torch.long).contiguous()
            dst = torch.tensor(dst, dtype=torch.long).contiguous()
            edge_index = torch.stack((src, dst))
            edge_index_w = share_edge(edge_index, sub_share_entity)
            new_edge_index_w = add_neighbor(uniq_entity,edge_index_w)
            X , Y = new_edge_index_w
            s.append(uniq_entity.tolist())
            edge_index_i.append(X.tolist())
            edge_index_j.append(Y.tolist())
            torch.cuda.empty_cache()
        except ValueError:
            pass
    s = np.array(s)
    edge_index_i = np.array(edge_index_i)
    edge_index_j = np.array(edge_index_j)
    '''
    return sent_features

if __name__ == '__main__':
    concept_graphs = pickle.load(open('utils/concept_graphs.pkl', 'rb'))
    all_seeds = pickle.load(open('utils/all_seeds.pkl', 'rb'))
    domains = ['books', 'kitchen']
    model = RGAT().cuda()
    ill = 0
    for d1 in domains:
         for d2 in domains:
            if d1 == d2:
                   continue
               #xsg = knowledge_graph(model,d1, split, all_seeds, concept_graphs)
               #np.save('knowledge_graph/knowledge_' + d1 + '_' + split + '_5000.npy',xsg)
               #读取两个域的知识图谱（三元组）
            '''
            xsg = np.load('knowledge_graph/knowledge_graph_' + d1 + '_' + split + '_5000.npy', allow_pickle=True)
            xtg = np.load('knowledge_graph/knowledge_graph_' + d2 + '_' + split + '_5000.npy', allow_pickle=True)
            #找到两个子图的共同邻居并构建邻接矩阵
            edge_index_ws = np.load('knowledge_graph/W_' + d1 + '_' + d2 + '_' + split + '_' + d1 + '_5000.npy', allow_pickle= True)
            edge_index_wt = np.load('knowledge_graph/W_' + d1 + '_' + d2 + '_' + split + '_' + d2 + '_5000.npy', allow_pickle=True)
            #针对去重后的实体给邻接矩阵重新编号
            data_s = add_neighbor(xsg,edge_index_ws)
            data_t = add_neighbor(xtg,edge_index_wt)
            #针对每个子图和共有邻居的邻接矩阵进行图嵌入
            sf_S = knowledge_graph(model, d1, split, all_seeds, concept_graphs, data_s)
            sf_T = knowledge_graph(model, d2, split, all_seeds, concept_graphs, data_t)
            #保存图嵌入
            np.save('share_grph_features/sf_' + d1 + '_' + d2 + '_' + split + '_' + d1 + '_5000.npy',sf_S)
            np.save('share_grph_features/sf_' + d1 + '_' + d2 + '_' + split + '_' + d2 + '_5000.npy', sf_T)
            #xsg = knowledge_graph(d1,split,all_seeds,concept_graphs)
            '''
            split = 'test'
            #xsg = np.load('knowledge_graph/knowledge_' + d1 + '_' + split + '_5000.npy',allow_pickle= True)
            #xtg = np.load('knowledge_graph/knowledge_' + d2 + '_' + split + '_5000.npy', allow_pickle=True)
            #share_entity = list(set(xsg.tolist()).intersection(set(xtg.tolist())))
            #entity_S , edge_index_si ,edge_index_sj = knowledge_graph(model, d1, split, all_seeds, concept_graphs, share_entity)
            entity_S = np.load('share_grph_features/entity_' + d1 + '_' + d2 + '_' + split + '_' + d1 + '_5000.npy',allow_pickle= True)
            edge_index_si =np.load('share_grph_features/edge_index_i_' + d1 + '_' + d2 + '_' + split + '_' + d1 + '_5000.npy'
                    ,allow_pickle= True)
            edge_index_sj = np.load('share_grph_features/edge_index_j_' + d1 + '_' + d2 + '_' + split + '_' + d1 + '_5000.npy',allow_pickle= True)
            entity_S = entity_S.tolist()
            edge_index_si = edge_index_si.tolist()
            edge_index_sj =  edge_index_sj.tolist()
            #x, dico = get_domain_dataset(d1, exp_type=split)
            sf_S = knowledge_graph(model,d1 , split, all_seeds, concept_graphs,entity_S,edge_index_si,edge_index_sj)
            np.save('share_grph_features/sf_' + d1 + '_' + d2 + '_' + split + '_' + d1 + '_5000.npy', sf_S)
            #entity_T , edge_index_ti ,edge_index_tj = knowledge_graph(model, d2, split, all_seeds, concept_graphs, share_entity)
            entity_T = np.load('share_grph_features/entity_' + d1 + '_' + d2 + '_' + split + '_' + d2 + '_5000.npy',allow_pickle= True )
            edge_index_ti =np.load('share_grph_features/edge_index_i_' + d1 + '_' + d2 + '_' + split + '_' + d2 + '_5000.npy',allow_pickle=True)
            edge_index_tj = np.load('share_grph_features/edge_index_j_' + d1 + '_' + d2 + '_' + split + '_' + d2 + '_5000.npy',allow_pickle=True)
            entity_T = entity_T.tolist()
            edge_index_ti = edge_index_ti.tolist()
            edge_index_tj = edge_index_tj.tolist()
            sf_T = knowledge_graph(model, d2, split, all_seeds, concept_graphs,entity_T,edge_index_ti,edge_index_tj)
            np.save('share_grph_features/sf_' + d1 + '_' + d2 + '_' + split + '_' + d2 + '_5000.npy', sf_T)
            #W_s = W_s.numpy()
            #W_t = W_t.numpy()

