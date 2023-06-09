import pickle

import spacy, os, numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from glove import read_pre_emb,get_r_emb
from gensim.corpora import Dictionary as gensim_dico
glv_countlist = []
#Word, embedding = read_pre_emb('glove.840B.300d.txt')
glv_countlist_word = []
z = []
nlp = spacy.load("en_core_web_sm")
kk = []
def parse_processed_amazon_dataset(FNames, emd,max_words):
    datasets = {}
    dico = gensim_dico()

    # First pass on document to build dictionary
    for fname in FNames:
        f = open(fname, encoding='utf-8')
        for l in f:
            tokens = l.split(sep=' ')
            label_string = tokens[-1]
            tokens_list = []
            for tok in tokens[:-1]:
                ts, tfreq = tok.split(':')
                freq = int(tfreq)
                tokens_list += [ts] * freq

            _ = dico.doc2bow(tokens_list, allow_update=True)

        f.close()

    # Preprocessing_options
    dico.filter_extremes(no_below=2)#不取前5000个而是取所有在频率为2以上的单词来构造一个词典
    dico.compactify()


    for fname in FNames:
        X = []
        Y = []

        docid = -1
        f = open(fname,encoding='utf-8')
        ill = 0
        for l in f:
            word_embedding = []
            ill = ill+1
            print(ill)
            tokens = l.split(sep=' ')
            label_string = tokens[-1]
            tokens_list = []
            for tok in tokens[:-1]:
                ts, tfreq = tok.split(':')
                freq = int(tfreq)
                tokens_list += [ts]*freq
            count_list = dico.doc2bow(tokens_list, allow_update=False)#根据字典的编号来排序的
            for i in count_list:
                x = i[0]
                freq = i[1]
                x1 = emd[x] * freq
                word_embedding.append(x1)
            word = torch.tensor(word_embedding)
            x = torch.mean(word,dim=0)

            #docid += 1

            X.append(x.view(-1,300).numpy().tolist())

            # Preprocess Label
            ls, lvalue = label_string.split(':')
            if ls == "#label#":
                if lvalue.rstrip() == 'positive':
                    lv = 1
                    Y.append(lv)
                elif lvalue.rstrip() == 'negative':
                    lv = 0
                    Y.append(lv)
                else:
                    raise Exception("Invalid Label Value")
            else:
                raise Exception('Invalid Format')

        datasets[fname] = (X, np.array(Y))
        f.close()
        del f

    return dico,datasets



def get_dataset_path(domain_name, exp_type):
    prefix ='./dataset/'
    if exp_type == 'small':
        fname = 'labelled.review'
    elif exp_type == 'all':
        fname = 'all.review'
    elif exp_type == 'test':
        fname = 'unlabeled.review'

    return os.path.join(prefix, domain_name, fname)



def get_dataset(target_name,split,emd, max_words=5000):
    """
    Returns source domain, target domain paired dataset
    """
    #source_path  = get_dataset_path(source_name, 'small')#small表示用于训练的源域和目标域
    target_path1 = get_dataset_path(target_name, split)
    #target_path2 = get_dataset_path(target_name, 'test')#test表示用于测试的目标域

    dataset_list = [target_path1]
    dico , datasets = parse_processed_amazon_dataset(dataset_list, emd,max_words)

    #X_s, Y_s = datasets[source_path]
    X_t1, Y_t1 = datasets[target_path1]
    #X_t2, Y_t2 = datasets[target_path2]

    #X_s  = count_list_to_sparse_matrix(L_s,  dico)
    #X_t1 = count_list_to_sparse_matrix(L_t1, dico)
    #X_t2 = count_list_to_sparse_matrix(L_t2, dico)

    return X_t1,Y_t1
if __name__ == '__main__':
    domains = [ 'books','dvd','electronics','kitchen']
    for d1 in domains:
        for split in ['test','small']:

            emd = np.load('dict/emd_' + d1 + '_'+ split+'.npy')
            X_t1, Y_t1 = get_dataset(d1, split,emd)
            X_t2 = np.array(X_t1)
            X_t3 = np.squeeze(X_t2)
            np.save('glove_features/glv_sf_' + d1 + '_' + split + '_5000.npy', X_t3)
            #np.save('glove_features/glv_label_' + d1 + '_' + split + '_5000.npy', Y_t1)

            '''
            word = np.load('dict/dict_' + d1 + '_' + split +'.npy')
            word1 = list(word)
            for x in word1:
                x1 = x.split(sep='_')
                z.append(x1)
            r_embedding = np.array(get_r_emb(z, Word, embedding))
            np.save('dict/emd_' + d1 + '_'+ split+'.npy',r_embedding)
            '''
            '''

            dico = get_dataset(d1,split)
            s = dico.token2id
            s1 = np.array(list(s.keys()))
            np.save('dict/dict_' + d1 + '_' + split +'.npy',s1)
            '''
            '''
            print('Extracting features for', d1, split)
            X_t1 , Y_t1 = get_dataset(d1,split)
            XZ_t1 = np.array(X_t1)
            np.save('glove_features/glv_label_' + d1 + '_' + split+'.npy',XZ_t1)
            np.ndarray.dump(X_t2, open('glove_features/glv_sf_' + d1 + '_' + 'test_5000.np', 'wb'))
            np.ndarray.dump(Y_t2, open('glove_features/glv_label_' + d1 + '_' + 'test_5000.np', 'wb'))
            '''

