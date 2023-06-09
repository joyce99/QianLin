#!/usr/bin/env python
# coding=utf-8

import os.path, pickle,wordninja
from tqdm import tqdm
x1 = []
z = []
z1 = []
relation_net = []


def list_add(a, b):
   c = []
   for i in range(len(a)):
      c.append(a[i] + b[i])
   return c
# 读取预训练的glove词嵌入
def read_pre_emb(file_path):
    words = []
    embeding = []
    print("read file:", file_path)
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        w_e = line.strip('\n').split(' ')
        words.append(w_e[0])
        emb = [float(e) for e in w_e[1:]]
        embeding.append(emb)
    return words, embeding



def get_r_emb(rs, words, emb, dim=300):
    r_emb = []
    il=0
    for r in tqdm(rs):
        embedding = [0 for n in range(dim)]
        if len(r) == 1:
            if r[0] in words:
                index = words.index(r[0])
                embedding = emb[index]
            else:
                index = words.index('unk')
                embedding = emb[index]
        if len(r) > 1:
            embedding = [0 for n in range(dim)]
            check = False
            for i in range(len(r)):
                if r[i] in words:
                    index = words.index(r[i])
                    embedding = list_add(embedding, emb[index ])
                    check = True
            if check == False:
                index = words.index('unk')
                embedding = emb[index]
            embedding = [e / float(len(r)) for e in embedding]
        r_emb.append(embedding)
        il = il +1
        if il % 10000 == 0:
            print(il)
    return r_emb


concept_net = pickle.load(open('utils/concepet1.pkl','rb'))
concept_map = pickle.load(open('utils/concept_map.pkl','rb'))
'''
relation_map = pickle.load(open('utils/relation_map.pkl','rb'))
for i in relation_map.keys():
    relation_net.append(i)
pickle.dump(relation_net,open('utils/relation_net.pkl','wb'))

#划分关系词
for y in relation_net:
    y1 = wordninja.split(y)
    z1.append(y1)
'''
#划分实体词
for x in concept_net:
    x1 = x.split(sep='_')
    z.append(x1)

word, embedding = read_pre_emb('glove.840B.300d.txt')  # 调用glove词向量库


r_embedding = get_r_emb(z, word, embedding)#取实体词向量
pickle.dump(r_embedding, open('utils/entity_embedding.pkl', 'wb'))
#relation_embedding = get_r_emb(z1,word,embedding)#取关系词向量
#pickle.dump(relation_embedding,open('utils/_FALE_relation_embedding.pkl','wb'))




