import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm
from torch_geometric.utils import *
from torch_scatter import scatter


  
class GCN(nn.Module):
    def __init__(self, hidden):
        super(GCN, self).__init__()
        self.w = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        deg = degree(edge_index_i, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index_j]*deg_inv_sqrt[edge_index_i]
        x = F.relu(spmm(edge_index[[1, 0]], norm, x.size(0), x.size(0), x))
        x = self.w(x)
        return x


class GCN_Rel(nn.Module):
    def __init__(self, h_hidden):
        super(GCN_Rel, self).__init__()
        # self.x_r = nn.Linear(100, 100, bias=False)

    def forward(self, x, edge_index, line_graph_val):
        edge_index_j, edge_index_i = edge_index
        deg = degree(edge_index_i, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index_j]*deg_inv_sqrt[edge_index_i]
        # rel = F.normalize(rel, dim=0)
        # alpha = self.x_r(x).squeeze()[edge_index_j] + self.x_r(x).squeeze()[edge_index_i]
        # alpha = softmax(F.leaky_relu(alpha), edge_index_j)
        # rel = softmax(F.leaky_relu(rel.float()), edge_index_j)
        # # norm = rel.float() + norm
        # norm = rel.float() + norm + alpha
        x = F.relu(spmm(edge_index[[1, 0]], norm, x.size(0), x.size(0), x))
        # x1 = F.relu(spmm(torch.cat([rel.view(1, -1), edge_index_j.view(1, -1)], dim=0), norm, rel.max()+1, x.size(0), x))
        # x2 = F.relu(spmm(torch.cat([rel.view(1, -1), edge_index_i.view(1, -1)], dim=0), norm, rel.max()+1, x.size(0), x))
        # x = self.x_r(x)
        return x


    
class Highway(nn.Module):
    def __init__(self, x_hidden):
        super(Highway, self).__init__()
        self.lin = nn.Linear(x_hidden, x_hidden)

    def forward(self, x1, x2):
        gate = torch.sigmoid(self.lin(x1))
        x = torch.mul(gate, x2)+torch.mul(1-gate, x1)
        return x


class GAT_E_to_R(nn.Module):
    def __init__(self, e_hidden, r_hidden):
        super(GAT_E_to_R, self).__init__()
        self.a_h1 = nn.Linear(r_hidden, 1, bias=False)
        self.a_h2 = nn.Linear(r_hidden, 1, bias=False)
        self.a_t1 = nn.Linear(r_hidden, 1, bias=False)
        self.a_t2 = nn.Linear(r_hidden, 1, bias=False)
        self.w_h = nn.Linear(e_hidden, r_hidden, bias=False)
        self.w_t = nn.Linear(e_hidden, r_hidden, bias=False)

        self.r_e1 = nn.Linear(e_hidden, 100, bias=False)
        self.r_e2 = nn.Linear(e_hidden, 100, bias=False)

    def forward(self, x_e, edge_index, rel, rel_emb, r_index, line_graph_index, line_graph_val):
        edge_index_h, edge_index_t = edge_index
        x_r_h = self.w_h(x_e)
        x_r_t = self.w_t(x_e)

        e1 = self.a_h1(x_r_h).squeeze()[edge_index_h]+self.a_h2(x_r_t).squeeze()[edge_index_t]
        e2 = self.a_t1(x_r_h).squeeze()[edge_index_h]+self.a_t2(x_r_t).squeeze()[edge_index_t]

        alpha = softmax(F.leaky_relu(e1).float(), rel)
        x_r_h = spmm(torch.cat([rel.view(1, -1), edge_index_h.view(1, -1)], dim=0), alpha, rel.max()+1, x_e.size(0), x_r_h)

        alpha = softmax(F.leaky_relu(e2).float(), rel)
        x_r_t = spmm(torch.cat([rel.view(1, -1), edge_index_t.view(1, -1)], dim=0), alpha, rel.max()+1, x_e.size(0), x_r_t)

        x_r = x_r_h+x_r_t
        return x_r

    
class GAT_R_to_E(nn.Module):
    def __init__(self, e_hidden, r_hidden):
        super(GAT_R_to_E, self).__init__()
        self.a_h = nn.Linear(e_hidden, 1, bias=False)
        self.a_t = nn.Linear(e_hidden, 1, bias=False)
        self.a_r = nn.Linear(r_hidden, 1, bias=False)


    def forward(self, x_e, x_r, edge_index, rel, line_graph_index, line_graph_val):
        edge_index_h, edge_index_t = edge_index
        e_h = self.a_h(x_e).squeeze()[edge_index_h]
        e_t = self.a_t(x_e).squeeze()[edge_index_t]
        e_r = self.a_r(x_r).squeeze()[rel]

        # line_graph_val = F.normalize(line_graph_val, dim=0, p=2)
        # e_l = softmax(line_graph_val, line_graph_index[0])
        # e_l = e_l + softmax(line_graph_val, line_graph_index[1])
        # # e_l = scatter(line_graph_val, line_graph_index[0])
        # e_l = e_l[rel]

        # alpha = softmax(F.leaky_relu(e_h+e_r+e_l).float(), edge_index_h)
        alpha = softmax(F.leaky_relu(e_h+e_r).float(), edge_index_h)
        x_e_h = spmm(torch.cat([edge_index_h.view(1, -1), rel.view(1, -1)], dim=0), alpha, x_e.size(0), x_r.size(0), x_r)
        # alpha = softmax(F.leaky_relu(e_t+e_r+e_l).float(), edge_index_t)
        alpha = softmax(F.leaky_relu(e_t+e_r).float(), edge_index_t)
        x_e_t = spmm(torch.cat([edge_index_t.view(1, -1), rel.view(1, -1)], dim=0), alpha, x_e.size(0), x_r.size(0), x_r)
        x = torch.cat([x_e_h, x_e_t], dim=1)
        return x


class GAT(nn.Module):
    def __init__(self, hidden, r_hidden):
        super(GAT, self).__init__()
        self.a_i = nn.Linear(hidden, 1, bias=False)
        self.a_j = nn.Linear(hidden, 1, bias=False)
        self.a_r = nn.Linear(r_hidden, 1, bias=False)
        
    def forward(self, x, r, edge_index, rel):
        edge_index_j, edge_index_i = edge_index
        e_i = self.a_i(x).squeeze()[edge_index_i]
        e_j = self.a_j(x).squeeze()[edge_index_j]
        r = self.a_r(r).squeeze()[rel]
        e = e_i+e_j+r
        alpha = softmax(F.leaky_relu(e).float(), edge_index_i)
        x = F.relu(spmm(edge_index[[1, 0]], alpha, x.size(0), x.size(0), x))
        return F.relu(x)


class GAT_R(nn.Module):
    def __init__(self, hidden):
        super(GAT_R, self).__init__()
        self.a_i = nn.Linear(hidden, 1, bias=False)
        self.a_j = nn.Linear(hidden, 1, bias=False)

    def forward(self, x, edge_index, line_graph_val):
        edge_index_j, edge_index_i = edge_index
        e_i = self.a_i(x).squeeze()[edge_index_i]
        e_j = self.a_j(x).squeeze()[edge_index_j]
        e = e_i + e_j
        # line_graph_val = F.normalize(line_graph_val, p=2, dim=0)
        # line_graph_val = F.relu(line_graph_val)
        alpha = softmax(F.leaky_relu(e).float(), edge_index_j)
        x = F.relu(spmm(edge_index[[1, 0]], alpha, x.size(0), x.size(0), x))
        return x



class GraphAttention(nn.Module):
    def __init__(self, e_hidden, r_hidden, depth=1, attn_heads=1):
        super(GraphAttention, self).__init__()
        self.e_hidden = e_hidden
        self.r_hidden = r_hidden
        self.depth = depth
        self.attn_heads = attn_heads
        self.attn_kernels = []

        self.attn_heads_reduction = 'avg'

        # self.ww1 = nn.Parameter(nn.init.sparse(torch.empty(300,1), sparsity=0.1))
        # self.ww2 = nn.Parameter(nn.init.sparse(torch.empty(300,1), sparsity=0.1))
        #
        self.ww1 = nn.Linear(300, 1, bias=False)
        self.ww2 = nn.Linear(300, 1, bias=False)
        self.attn_kernels.append([])
        # for l in range(self.depth):
        #     self.attn_kernels.append([])
        #     for head in range(self.attn_heads):
        # w = torch.Tensor(e_hidden, 1)
        # nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu')).requires_grad_()
        self.attn_kernels[0].append(self.ww1)
        self.attn_kernels.append([])
        self.attn_kernels[1].append(self.ww2)
        self.x = nn.Linear(500, 100, bias=False)
        self.r_emb = nn.Linear(300, 100, bias=False)
        self.x_r = nn.Linear(300, 100, bias=False)

    def forward(self, x, edge_index_all, rel_emb, r_index, line_graph_index, line_graph_val):
        outputs = []
        x = self.x(x)
        features = x
        features = F.leaky_relu(features)
        outputs.append(features)

        r_val = torch.ones(r_index[0].size(0)).to('cuda')
        r_val = softmax(r_val, r_index[0])

        rel_emb = self.r_emb(rel_emb)
        rel_emb = torch.cat([rel_emb, rel_emb], dim=0)

        adj1, adj2 = torch.unique(edge_index_all, dim=1)
        for l in range(self.depth):
            features_list = []
            for head in range(self.attn_heads):
                attention_kernel = self.attn_kernels[l][head]
                rels_sum = spmm(r_index[[0, 1]], r_val, adj1.size(0), rel_emb.size(0), rel_emb).to('cuda')
                # rels_sum = spmm(r_index[[0, 1]], r_val, adj1.size(0), rel_emb.size(0), rel_emb)
                rels_sum = F.normalize(rels_sum, p=2, dim=1)

                neighs = features[adj2]
                selfs = features[adj1]

                bias = torch.sum(neighs * rels_sum, dim=1, keepdim=True) * rels_sum
                neighs = neighs - 2 * bias

                att = attention_kernel(torch.cat([selfs, neighs, rels_sum], dim=1)).squeeze()
                # att = torch.mm(torch.cat([selfs, neighs, rels_sum], dim=1), attention_kernel).squeeze()
                att = softmax(att, adj1)

                new_features = scatter(neighs * torch.unsqueeze(att, dim=-1), adj1, dim=0, reduce='sum')
                features_list.append(new_features)

            if self.attn_heads_reduction == 'concat':
                features = torch.cat(features_list, dim=1)
            else:
                features = torch.mean(torch.stack(features_list), dim=0)

            features = F.leaky_relu(features)
            outputs.append(features)

        outputs = torch.cat(outputs, dim=1)
        return outputs


class GraphAttention_wjq(nn.Module):
    def __init__(self, e_hidden, r_hidden):
        super(GraphAttention_wjq, self).__init__()
        self.e_hidden = e_hidden
        self.r_hidden = r_hidden
        # self.ww1 = nn.Parameter(nn.init.sparse(torch.empty(300,1), sparsity=0.1))
        # self.ww2 = nn.Parameter(nn.init.sparse(torch.empty(300,1), sparsity=0.1))
        self.ww1 = nn.Linear(700, 1, bias=False)
        # self.ww2 = nn.Linear(400, 1, bias=False)
        # self.attn_kernels = []
        # self.attn_kernels.append(self.ww1)
        # self.attn_kernels.append(self.ww2)

    def forward(self, x, edge_index_all, rel_all, rel_emb):
        outputs = []
        # x = self.x1(x)
        e_features = F.leaky_relu(x)
        edge_index_i, edge_index_j = edge_index_all

        e_head = e_features[edge_index_i]
        e_tail = e_features[edge_index_j]

        #
        # r_features = F.leaky_relu(rel_emb)
        # edge_index_rel_i, edge_index_rel_j = line_graph_index
        #
        # line_graph_val = F.normalize(line_graph_val, p=2, dim=0)
        # line_graph_val1 = softmax(line_graph_val, edge_index_rel_i).view(-1, 1)
        #
        # line_graph_val2 = softmax(line_graph_val, edge_index_rel_j).view(-1, 1)
        #
        # r_head = r_features[edge_index_rel_i]
        # r_tail = r_features[edge_index_rel_j]
        # r1 = scatter(r_head * line_graph_val1, edge_index_rel_i, dim=0, reduce='sum')
        # r2 = scatter(r_tail * line_graph_val2, edge_index_rel_j, dim=0, reduce='sum')

        # rel_emb_x = torch.cat([rel_emb, rel_emb], dim=0)
        #
        rel_emb = F.leaky_relu(rel_emb)
        e_rel = rel_emb[rel_all]


        # for i in range(2):
        # att = self.attn_kernels[0]
        # att = self.ww1(torch.cat([e_head, e_tail], dim=1)).squeeze()
        att = self.ww1(torch.cat([e_head, e_rel, e_tail], dim=1)).squeeze()
        att = softmax(att, edge_index_i)

        x_e = scatter(torch.cat([e_head, e_rel, e_tail], dim=1) * torch.unsqueeze(att, dim=-1), edge_index_i, dim=0, reduce='sum')
        # x_e = scatter(torch.cat([e_head, e_tail], dim=1) * torch.unsqueeze(att, dim=-1), edge_index_i, dim=0, reduce='sum')
        outputs.append(x_e)

        return torch.cat(outputs, dim=1)

        return torch.mean(torch.stack(outputs, dim=1), dim=1)





        #
        # rel_emb = self.r_emb(rel_emb)
        # x_r = self.x_r(x_r)
        # rel_emb = torch.cat([x_r, rel_emb], dim=0)
        #
        # adj1, adj2 = torch.unique(edge_index_all, dim=1)
        # for l in range(self.depth):
        #     features_list = []
        #     for head in range(self.attn_heads):
        #         attention_kernel = self.attn_kernels[l][head]
        #         rels_sum = spmm(r_index[[0, 1]], r_val, adj1.size(0), rel_emb.size(0), rel_emb).to('cuda')
        #         # rels_sum = spmm(r_index[[0, 1]], r_val, adj1.size(0), rel_emb.size(0), rel_emb)
        #         rels_sum = F.normalize(rels_sum, p=2, dim=1)
        #
        #         neighs = features[adj2]
        #         selfs = features[adj1]
        #
        #         bias = torch.sum(neighs * rels_sum, dim=1, keepdim=True) * rels_sum
        #         neighs = neighs - 2 * bias
        #
        #         att = attention_kernel(torch.cat([selfs, neighs, rels_sum], dim=1)).squeeze()
        #         # att = torch.mm(torch.cat([selfs, neighs, rels_sum], dim=1), attention_kernel).squeeze()
        #         att = softmax(att, adj1)
        #
        #         new_features = scatter(neighs * torch.unsqueeze(att, dim=-1), adj1, dim=0, reduce='sum')
        #         features_list.append(new_features)
        #
        #     if self.attn_heads_reduction == 'concat':
        #         features = torch.cat(features_list, dim=1)
        #     else:
        #         features = torch.mean(torch.stack(features_list), dim=0)
        #
        #     features = F.leaky_relu(features)
        #     outputs.append(features)
        #
        # outputs = torch.cat(outputs, dim=1)
        return x_e



class RAGA(nn.Module):
    def __init__(self, e_hidden, r_hidden, rel1_size, rel2_size):
        super(RAGA, self).__init__()
        self.gcn1 = GCN(e_hidden)
        self.highway1 = Highway(e_hidden)
        self.gcn2 = GCN(e_hidden)
        self.highway2 = Highway(e_hidden)
        # self.gcn_rel = GCN_Rel(e_hidden)

        # self.gat_e_to_r = GAT_E_to_R(e_hidden, r_hidden)
        self.graphattention_wjq = GraphAttention_wjq(e_hidden, r_hidden)
        # self.graphattention = GraphAttention(e_hidden, r_hidden)
        # self.gat_r_to_e = GAT_R_to_E(e_hidden, r_hidden)
        # self.gat = GAT(e_hidden+2*r_hidden)
        self.gat = GAT(1000, r_hidden)
        self.gat_r = GAT_R(100)
        self.rel_emb1 = nn.Parameter(nn.init.sparse_(torch.empty(rel1_size, r_hidden), sparsity=0.15))
        self.rel_emb2 = nn.Parameter(nn.init.sparse_(torch.empty(rel2_size, r_hidden), sparsity=0.15))
        # self.rel_emb = nn.Parameter(nn.init.xavier_normal(torch.empty(1701, 100)))
        # self.rel_emb = nn.Parameter(nn.init.xavier_normal(torch.Tensor(1701, 100)))

    # def forward(self, x_e, edge_index, rel, edge_index_all, rel_all, rel_emb, r_index, line_graph_index, line_graph_val):
    def forward(self, x_e, edge_index, rel, edge_index_all, rel_all, line_graph_index_out, line_graph_val_out, line_graph_index_in, line_graph_val_in):
        # edge_index_all = add_self_loops(edge_index_all)[0]
        x_e = self.highway1(x_e, self.gcn1(x_e, edge_index_all))
        x_e = self.highway2(x_e, self.gcn2(x_e, edge_index_all))


        # x_r = self.gat_e_to_r(x_e, edge_index, rel, rel_emb, r_index, line_graph_index, line_graph_val)

        # rel_emb = self.gcn_rel(self.rel_emb, line_graph_index, line_graph_val)

        if rel.max()+1 == self.rel_emb1.size(0):
            rel_emb_out = self.gat_r(self.rel_emb1, line_graph_index_out, line_graph_val_out)
            rel_emb_in = self.gat_r(self.rel_emb1, line_graph_index_in, line_graph_val_in)
            # rel_emb_out = self.rel_emb1
            # rel_emb_in = self.rel_emb1
        else:
            rel_emb_out = self.gat_r(self.rel_emb2, line_graph_index_out, line_graph_val_out)
            rel_emb_in = self.gat_r(self.rel_emb2, line_graph_index_in, line_graph_val_in)
            # rel_emb_out = self.rel_emb2
            # rel_emb_in = self.rel_emb2

        rel_emb = torch.cat([rel_emb_out, rel_emb_in], dim=0)


        # x_r = torch.cat([rel_emb, self.gat_e_to_r(x_e, edge_index, rel)], dim=1)
        #
        # x_e = torch.cat([x_e, self.gat_r_to_e(x_e, x_r, edge_index, rel, line_graph_index, line_graph_val)],dim=1)

        # edge_index_all = remove_self_loops(edge_index_all)[0]

        # x_e = torch.cat([x_e, self.graphattention(x_e, edge_index_all, rel_emb, r_index, line_graph_index, line_graph_val)], dim=1)
        # x_wjq = torch.cat([x_e, self.graphattention_wjq(x_e, edge_index_all, rel_all, rel_emb, r_index, line_graph_index, line_graph_val)], dim=1)
        x_wjq = torch.cat([x_e, self.graphattention_wjq(x_e, edge_index_all, rel_all, rel_emb)], dim=1)
        # x_wjq = torch.cat([x_e, self.graphattention_wjq(x_e, edge_index_all, rel_all, self.rel_emb, r_index, line_graph_index, line_graph_val)], dim=1)

        # x_e_emb = torch.mean(torch.stack([x_e_emb, x_r_emb]), dim=0)
        # x_e = torch.cat([x_e, x_g_emb, x_e_emb, x_r_emb], dim=1)
        edge_index_all, rel_all = remove_self_loops(edge_index_all, rel_all)
        # x_e = torch.cat([x_e_emb, self.gat(x_e_emb, edge_index_all)], dim=1)
        # x_e = torch.cat([x_wjq, self.gat(x_wjq, edge_index_all)], dim=1)
        x_e = torch.cat([x_wjq, self.gat(x_wjq, rel_emb, edge_index_all, rel_all)], dim=1)
        return x_e
