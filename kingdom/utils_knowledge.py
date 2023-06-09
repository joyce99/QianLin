import warnings

warnings.filterwarnings('ignore')  # 警告扰人，手动封存
from tqdm import tqdm
from utils_graph import unique_rows
from utils import get_domain_dataset, spacy_seed_concepts_list
import numpy as np, pickle, argparse

import torch
import torch.nn.functional as F
from rgcn import RGAT
from torch_scatter import scatter_add
from torch_geometric.data import Data
def knowledge_graph( domain, split, all_seeds, concept_graphs):
    """
        Graph features for each sentence (document) instance in a domain.
    """
    x, dico = get_domain_dataset(domain, exp_type=split)
    d = list(dico.values())
    sent_features = np.zeros((len(x), 100))
    glv_sent_features = np.zeros((len(x), 100))

    for j in tqdm(range(len(x)), position=0, leave=False):
        c = [dico.id2token[item] for item in np.where(x[j] != 0)[0]]
        n = list(spacy_seed_concepts_list(c).intersection(set(all_seeds)))
        try:
            xg = np.concatenate([concept_graphs[item] for item in n])  # 内存链接在一起
            xg = xg[~np.all(xg == 0, axis=1)]

            # absent1 = set(xg[:, 0]) - unique_nodes_mapping.keys()
            # absent2 = set(xg[:, 2]) - unique_nodes_mapping.keys()
            # absent = absent1.union(absent2)

            # for item in absent:
            #   xg = xg[~np.any(xg == item, axis=1)]

            # xg[:, 0] = np.vectorize(unique_nodes_mapping.get)(xg[:, 0])
            # xg[:, 2] = np.vectorize(unique_nodes_mapping.get)(xg[:, 2])
            xg = unique_rows(xg).astype('int64')
            if len(xg) > 50000:
                xg = xg[:50000, :]


        except ValueError:
            pass

    return xg
if __name__ == '__main__':
    all_seeds = pickle.load(open('utils/all_seeds.pkl', 'rb'))
    concept_graphs = pickle.load(open('utils/concept_graphs.pkl', 'rb'))
    domains = ['books', 'dvd', 'electronics', 'kitchen']
    for d1 in domains:
        for split in ['test', 'small']:
            print('Extracting knowledge_graph for', d1, split)
            XSG = knowledge_graph(d1, split, all_seeds, concept_graphs)
            np.ndarray.dump(XSG, open('knowledge_graph/glv_sf_' + d1 + '_' +split+ '_5000.np', 'wb'))
