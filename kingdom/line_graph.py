# 处理KG1得到其关系三元组结点图的对应线图，KG2同理
from torch_geometric.io import read_txt_array
import torch
from torch_geometric.utils import sort_edge_index
import numpy as np
import itertools



def process():
    x1_path = './data/DBP15K/zh_en/ent_ids_1'
    x2_path = './data/DBP15K/zh_en/ent_ids_2'
    g1_path = './data/DBP15K/zh_en/triples_1'
    g2_path = './data/DBP15K/zh_en/triples_2'

    print('process KG1')
    # line_graph1 = process_graph(g1_path, x1_path)
    line_graph1_out, line_graph1_in = process_graph(g1_path, x1_path)
    np.save('./data/DBP15K/zh_en/line_graph1_v3_out.npy', line_graph1_out)
    np.save('./data/DBP15K/zh_en/line_graph1_v3_in.npy', line_graph1_in)
    # check_Ring(g1_path, x1_path)


    print('process KG2')
    # line_graph2 = process_graph(g2_path, x2_path)
    line_graph2_out, line_graph2_in = process_graph(g2_path, x2_path)
    np.save('./data/DBP15K/zh_en/line_graph2_v3_out.npy', line_graph2_out)
    np.save('./data/DBP15K/zh_en/line_graph2_v3_in.npy', line_graph2_in)
    # check_Ring(g2_path, x2_path)

def process_graph(triple_path, ent_path):
    g = read_txt_array(triple_path, sep='\t', dtype=torch.long)
    subj, rel, obj = g.t()#头实体尾实体和关系


    assoc = torch.full((rel.max().item() + 1,), -1, dtype=torch.long)#构造一个rel.max+1的值为-1的矩阵
    assoc[rel.unique()] = torch.arange(rel.unique().size(0))
    rel = assoc[rel]

    idx = []
    with open(ent_path, 'r') as f:
        for line in f:
            info = line.strip().split('\t')
            idx.append(int(info[0]))
    idx = torch.tensor(idx)

    assoc = torch.full((idx.max().item() + 1,), -1, dtype=torch.long)
    assoc[idx] = torch.arange(idx.size(0))
    subj, obj = assoc[subj], assoc[obj]
    edge_index1 = torch.stack([subj, obj], dim=0)
    edge_index1, rel1 = sort_edge_index(edge_index1, rel)
    edge_index1, rel1 = edge_index1.numpy().tolist(), rel1.numpy().tolist()

    edge_index2 = torch.stack([obj, subj], dim=0)
    edge_index2, rel2 = sort_edge_index(edge_index2, rel)
    edge_index2, rel2 = edge_index2.numpy().tolist(), rel2.numpy().tolist()

    print('process out')
    # line_graph_out = get_line_graph_v1(edge_index1, rel1)
    # line_graph_out = get_line_graph_v2(edge_index1, rel1, out=True)
    line_graph_out = get_line_graph_v3(edge_index1, rel1, out=True)
    print('process in')
    # line_graph_in = get_line_graph_v1(edge_index2, rel2)
    # line_graph_in = get_line_graph_v2(edge_index2, rel2, out=False)
    line_graph_in = get_line_graph_v3(edge_index2, rel2, out=False)
    # line_graph = line_graph_out + line_graph_in
    return line_graph_out, line_graph_in
    # return line_graph


# 第一个版本的获取线图的方式：将出度和入度分开处理，最后将两个矩阵进行相加 最终矩阵大小:rel_size×rel_size
def get_line_graph_v1(edge_index, rel):
    line_graph = np.zeros((max(rel)+1, max(rel)+1))
    length = 0
    for i in range(max(edge_index[0])+1):
        rel_index = []
        j = 0
        while (j + length < len(edge_index[0]) and edge_index[0][j + length] == i):
            rel_index.append(rel[j + length])
            length += 1
        print(i)
        for index in itertools.combinations(rel_index, 2):
            line_graph[index[0]][index[1]] += 1
            line_graph[index[1]][index[0]] += 1
    return line_graph


# 第二个版本的获取线图的方式：将出度和入度分开处理，最后将两个矩阵就行对角拼接，最终矩阵大小:2*rel_size×2*rel_size
def get_line_graph_v2(edge_index, rel, out):
    size = max(rel)+1
    line_graph = np.zeros((2*size, 2*size))
    length = 0
    for i in range(max(edge_index[0])+1):
        rel_index = []
        j = 0
        while (j + length < len(edge_index[0]) and edge_index[0][j + length] == i):
            rel_index.append(rel[j + length])
            length += 1
        print(i)
        if not out:
            for index in itertools.combinations(rel_index, 2):
                line_graph[index[0]+size][index[1]+size] += 1
                line_graph[index[1]+size][index[0]+size] += 1
        else:
            for index in itertools.combinations(rel_index, 2):
                line_graph[index[0]][index[1]] += 1
                line_graph[index[1]][index[0]] += 1
    return line_graph



# 第三个版本的获取线图的方式：将出度和入度分开处理，分别得到两个out和in矩阵大小:rel_size×rel_size，矩阵中的值是(1/该结点的出度)
def get_line_graph_v3(edge_index, rel):
    line_graph = np.zeros((max(rel)+1, max(rel)+1))
    length = 0
    for i in range(max(edge_index[0])+1):
        rel_index = []
        j = 0
        while (j + length < len(edge_index[0]) and edge_index[0][j + length] == i):
            rel_index.append(rel[j + length])
            length += 1
        print(i)
        for index in itertools.combinations(rel_index, 2):
            tt = 1/(len(rel_index))
            line_graph[index[0]][index[1]] += 1/(len(rel_index))
            line_graph[index[1]][index[0]] += 1/(len(rel_index))
    return line_graph


def add_inverse_rels(edge_index):
    edge_index_all = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    return edge_index_all



# en存在三角结构的结点数量：13986 总实体数量：19572
# zh存在三角结构的结点数量：11709 总实体数量：19388



def check_Ring(triple_path, ent_path):
    g = read_txt_array(triple_path, sep='\t', dtype=torch.long)
    subj, rel, obj = g.t()

    assoc = torch.full((rel.max().item() + 1,), -1, dtype=torch.long)
    assoc[rel.unique()] = torch.arange(rel.unique().size(0))
    rel = assoc[rel]

    idx = []
    with open(ent_path, 'r') as f:
        for line in f:
            info = line.strip().split('\t')
            idx.append(int(info[0]))
    idx = torch.tensor(idx)

    assoc = torch.full((idx.max().item() + 1,), -1, dtype=torch.long)
    assoc[idx] = torch.arange(idx.size(0))
    subj, obj = assoc[subj], assoc[obj]
    edge_index = torch.stack([subj, obj], dim=0)
    edge_index, rel = sort_edge_index(edge_index, rel)
    edge_index_all = add_inverse_rels(edge_index)
    edge_index_all, _ = sort_edge_index(edge_index_all)

    edge_index_all_i, edge_index_all_j = edge_index_all.numpy()

    #-----------第一层 邻居

    length = 0
    neighs_hop1 = dict()
    for i in range(max(edge_index_all_i)+1):
        rel_index = []
        j = 0
        while(j+length < len(edge_index_all_i) and edge_index_all_i[j+length] == i):
            rel_index.append(edge_index_all_j[j+length])
            length += 1
        if i % 3000 == 0:
            print(i)
        neighs_hop1[i] = set(rel_index)

    ring_hop1 = []
    for key, value in neighs_hop1.items():
        if key in value:
            ring_hop1.append(key)

    print('存在self_loop的结点数量'+str(len(ring_hop1)))
    #-----------第二层 邻居的邻居

    neighs_hop2_selfloop = dict()
    neighs_hop2_remove_selfloop = dict()
    for i in range(len(neighs_hop1)):
        neighs_hop2_selfloop[i] = set()
        neighs_hop2_remove_selfloop[i] = set()
        for neigh in neighs_hop1[i]:
            xx = neighs_hop1[neigh].copy()
            if i in xx:
                xx.remove(i)
            neighs_hop2_selfloop[i] = (neighs_hop2_selfloop[i] | xx)
            neighs_hop2_remove_selfloop[i] = (neighs_hop2_remove_selfloop[i] | xx)
        if i in neighs_hop2_remove_selfloop[i]:
            neighs_hop2_remove_selfloop[i].remove(i)
        if i % 3000 == 0:
            print(i)

    ring_hop2_selfloop = []
    for key, value in neighs_hop2_selfloop.items():
        if key in value:
            ring_hop2_selfloop.append(key)

    ring_hop2_remove_selfloop = []
    for key, value in neighs_hop2_remove_selfloop.items():
        if key in value:
            ring_hop2_remove_selfloop.append(key)


    #-----------第三层邻居 寻找三层的邻居中有没有初始的结点，有的话则表示该初始结点存在三角结构

    neighs_hop3 = dict()
    for i in range(len(neighs_hop2_remove_selfloop)):
        neighs_hop3[i] = set()
        for neigh in neighs_hop2_remove_selfloop[i]:
            neighs_hop3[i] = (neighs_hop3[i] | neighs_hop1[neigh])
        if i % 3000 == 0:
            print(i)

    ring_hop3 = []
    for key, value in neighs_hop3.items():
        if key in value:
            ring_hop3.append(key)

    print('存在三角结构的结点数量'+str(len(ring_hop3)))

    return

process()
